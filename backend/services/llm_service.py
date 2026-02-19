"""
LLM integration for advanced mapping decisions using OpenRouter API.

This module provides LLM-based reasoning for schema mapping:
- Calls OpenRouter API (OpenAI-compatible) with structured prompts
- Disambiguates between similar candidates
- Provides confidence scoring based on LLM output and heuristics
- Falls back gracefully if API is unavailable
- Implements rate limiting and retry logic
- Caches responses to reduce duplicate API calls
"""
import json
import requests
import time
import hashlib
from typing import Dict, Any, Optional, List
from threading import Lock
from collections import deque
from datetime import datetime, timedelta

from backend.config import (
    LLM_API_URL, LLM_API_KEY, LLM_MODEL, DISABLE_SSL_VERIFY,
    LLM_MAX_RPM, LLM_MAX_RPD, LLM_MAX_RETRIES, LLM_RETRY_DELAY,
    LLM_MAX_WAIT_TIME, ENABLE_LLM_CACHE
)
from backend.models import ColumnProfile, CandidateScore, MemoryRecord


# ============================================================================
# Global Rate Limiter
# ============================================================================

class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    
    Implements both per-minute and per-day rate limits with thread safety.
    """
    
    def __init__(self, max_rpm: int, max_rpd: int):
        self.max_rpm = max_rpm
        self.max_rpd = max_rpd
        
        # Track requests in sliding windows
        self.minute_requests = deque()
        self.day_requests = deque()
        
        # Thread safety
        self.lock = Lock()
    
    def _clean_old_requests(self):
        """Remove requests older than the time windows."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        day_ago = now - timedelta(days=1)
        
        # Clean minute window
        while self.minute_requests and self.minute_requests[0] < minute_ago:
            self.minute_requests.popleft()
        
        # Clean day window
        while self.day_requests and self.day_requests[0] < day_ago:
            self.day_requests.popleft()
    
    def can_make_request(self) -> bool:
        """Check if a request can be made without exceeding limits."""
        with self.lock:
            self._clean_old_requests()
            return (len(self.minute_requests) < self.max_rpm and 
                   len(self.day_requests) < self.max_rpd)
    
    def wait_if_needed(self, max_wait: float = None) -> bool:
        """
        Wait until a request can be made or timeout is reached.
        
        Args:
            max_wait: Maximum time to wait in seconds
        
        Returns:
            True if request can proceed, False if timed out
        """
        max_wait = max_wait or LLM_MAX_WAIT_TIME
        start_time = time.time()
        
        while not self.can_make_request():
            elapsed = time.time() - start_time
            if elapsed >= max_wait:
                return False
            
            # Calculate wait time based on next available slot
            with self.lock:
                self._clean_old_requests()
                
                # Check minute limit
                if len(self.minute_requests) >= self.max_rpm:
                    oldest = self.minute_requests[0]
                    wait_until = oldest + timedelta(minutes=1)
                    wait_seconds = (wait_until - datetime.now()).total_seconds()
                    wait_seconds = max(0.1, min(wait_seconds, max_wait - elapsed))
                    print(f"Rate limit: waiting {wait_seconds:.1f}s (minute quota)")
                    time.sleep(wait_seconds)
                    continue
                
                # Check day limit
                if len(self.day_requests) >= self.max_rpd:
                    print(f"Rate limit: daily quota exceeded ({self.max_rpd} requests)")
                    return False
                
                # Small delay to avoid tight loop
                time.sleep(0.1)
        
        return True
    
    def record_request(self):
        """Record that a request was made."""
        with self.lock:
            now = datetime.now()
            self.minute_requests.append(now)
            self.day_requests.append(now)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current rate limiter statistics."""
        with self.lock:
            self._clean_old_requests()
            return {
                "requests_last_minute": len(self.minute_requests),
                "requests_last_day": len(self.day_requests),
                "minute_limit": self.max_rpm,
                "day_limit": self.max_rpd,
                "minute_available": self.max_rpm - len(self.minute_requests),
                "day_available": self.max_rpd - len(self.day_requests)
            }


# Global rate limiter instance
_rate_limiter = RateLimiter(max_rpm=LLM_MAX_RPM, max_rpd=LLM_MAX_RPD)

# ============================================================================
# Response Cache
# ============================================================================

class LLMResponseCache:
    """Simple in-memory cache for LLM responses to avoid duplicate API calls."""
    
    def __init__(self):
        self.cache = {}
        self.lock = Lock()
    
    def _hash_prompt(self, prompt: str) -> str:
        """Generate hash key for a prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]
    
    def get(self, prompt: str) -> Optional[Dict]:
        """Get cached response for a prompt."""
        if not ENABLE_LLM_CACHE:
            return None
        
        key = self._hash_prompt(prompt)
        with self.lock:
            return self.cache.get(key)
    
    def set(self, prompt: str, response: Dict):
        """Cache a response for a prompt."""
        if not ENABLE_LLM_CACHE:
            return
        
        key = self._hash_prompt(prompt)
        with self.lock:
            self.cache[key] = response
    
    def clear(self):
        """Clear all cached responses."""
        with self.lock:
            self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                "cached_responses": len(self.cache),
                "enabled": ENABLE_LLM_CACHE
            }


# Global cache instance
_response_cache = LLMResponseCache()


def call_llm_api(prompt: str, model: str = None, retry: bool = True) -> Optional[Dict]:
    """
    HTTP call to OpenRouter API with rate limiting, retry logic, and caching.

    OpenRouter is OpenAI-compatible; uses same chat completions format.
    Handles authentication, rate limiting, retries with exponential backoff,
    and caches responses to avoid duplicate calls.

    Args:
        prompt: The prompt text to send to the LLM
        model: Optional model id (defaults to OPENROUTER_MODEL from config)
        retry: Whether to retry on failure (default: True)

    Returns:
        Dictionary with LLM response, or None if API call fails
    """
    if not LLM_API_KEY:
        print("Warning: OPENROUTER_API_KEY not configured. LLM integration disabled.")
        return None

    if not LLM_API_URL:
        print("Warning: OPENROUTER_API_URL not configured. LLM integration disabled.")
        return None

    # Check cache first
    cached_response = _response_cache.get(prompt)
    if cached_response:
        print("✓ Using cached LLM response")
        return cached_response

    if model is None:
        model = LLM_MODEL

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/mapper-phase3",  # optional, for OpenRouter
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a schema mapping expert. Analyze source columns and target fields to recommend the best mappings. Always respond with valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 500
    }

    # Retry loop with exponential backoff
    max_retries = LLM_MAX_RETRIES if retry else 1
    retry_delay = LLM_RETRY_DELAY
    attempt = 0                  # counts non-429 attempts only
    rate_limit_start = None      # tracks total wall-clock time spent on 429 waits
    
    while True:
        try:
            # Wait for rate limiter before making request
            if not _rate_limiter.wait_if_needed():
                print("Error: Rate limit quota exceeded, cannot make request")
                return None
            
            # Record request in rate limiter
            _rate_limiter.record_request()
            
            # Make API call (OpenRouter is OpenAI-compatible)
            response = requests.post(
                LLM_API_URL,
                headers=headers,
                json=payload,
                timeout=30,
                verify=not DISABLE_SSL_VERIFY,
            )
            response.raise_for_status()

            result = response.json()

            # Cache successful response
            _response_cache.set(prompt, result)

            stats = _rate_limiter.get_stats()
            print(f"✓ OpenRouter API call successful (RPM: {stats['requests_last_minute']}/{stats['minute_limit']})")

            return result

        except requests.exceptions.Timeout:
            attempt += 1
            if attempt < max_retries:
                print(f"Warning: OpenRouter API timeout (attempt {attempt}/{max_retries}), retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            print("Error: OpenRouter API request timed out after all retries")
            return None
        
        except requests.exceptions.HTTPError as e:
            # IMPORTANT: use "is not None" — Response.__bool__() returns False for 4xx/5xx
            status_code = e.response.status_code if e.response is not None else 0
            
            # Parse error details
            try:
                err_body = e.response.json() if e.response is not None else {}
                err_msg = err_body.get("error", {}).get("message", str(e))
            except Exception:
                err_msg = str(e)
            
            # Handle 429 (rate limit) — always retry until wall-clock timeout
            if status_code == 429:
                now = time.time()
                if rate_limit_start is None:
                    rate_limit_start = now
                
                elapsed = now - rate_limit_start
                if elapsed >= LLM_MAX_WAIT_TIME:
                    print(f"✗ Rate limit: gave up after waiting {elapsed:.1f}s (max {LLM_MAX_WAIT_TIME}s)")
                    print(f"  Current stats: {_rate_limiter.get_stats()}")
                    return None

                wait_time = retry_delay
                retry_after = e.response.headers.get('retry-after')
                if retry_after:
                    try:
                        wait_time = float(retry_after)
                    except ValueError:
                        pass

                remaining = LLM_MAX_WAIT_TIME - elapsed
                wait_time = min(wait_time, remaining)

                print(f"⚠ Rate limited (429). Waiting {wait_time:.1f}s before retry "
                      f"(elapsed {elapsed:.1f}s / {LLM_MAX_WAIT_TIME}s)...")
                time.sleep(wait_time)
                retry_delay = min(retry_delay * 2, 30)  # Exponential backoff, cap at 30s
                continue   # does NOT increment `attempt`
            
            # Handle other HTTP errors
            attempt += 1
            if attempt < max_retries and status_code >= 500:
                print(f"Warning: OpenRouter API server error {status_code} (attempt {attempt}/{max_retries}), retrying...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue

            print(f"Error: OpenRouter API returned HTTP {status_code}: {err_msg}")
            return None

        except requests.exceptions.RequestException as e:
            attempt += 1
            if attempt < max_retries:
                print(f"Warning: OpenRouter API request failed (attempt {attempt}/{max_retries}), retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            print(f"Error: OpenRouter API request failed after all retries: {e}")
            return None

        except json.JSONDecodeError:
            print("Error: Failed to parse OpenRouter API response as JSON")
            return None


def build_llm_prompt(source_profile: ColumnProfile, candidates: List[Dict], 
                     memory_examples: List[Dict] = None) -> str:
    """
    Construct structured prompt with source column details, shortlisted candidates, and memory examples.
    
    Builds a comprehensive prompt that includes:
    - Source column name, type, and sample values
    - Candidate target fields with descriptions
    - Historical mapping examples from memory
    
    Args:
        source_profile: ColumnProfile object with source column details
        candidates: List of candidate dicts with 'target' and 'score' keys
        memory_examples: Optional list of memory search results
    
    Returns:
        Formatted prompt string for the LLM
    """
    
    # Build candidate list
    candidates_text = ""
    for i, candidate in enumerate(candidates, 1):
        target = candidate.get('target')
        score = candidate.get('score', 0.0)
        
        # Handle different candidate formats
        if hasattr(target, 'field_name'):
            # TargetField object
            field_name = target.field_name
            description = target.description or "No description"
            category = target.category or "N/A"
            required = target.required
        elif isinstance(target, dict):
            # Dict format
            field_name = target.get('field_name', 'Unknown')
            description = target.get('description', 'No description')
            category = target.get('category', 'N/A')
            required = target.get('required', False)
        else:
            continue
        
        candidates_text += f"""{i}. {field_name}
   - Description: {description}
   - Category: {category}
   - Required: {required}
   - Heuristic Score: {score:.2f}

"""
    
    # Build historical examples section
    examples_text = ""
    if memory_examples:
        examples_text = "\nHistorical Mapping Examples:\n"
        for example in memory_examples[:5]:  # Show top 5 examples
            record = example.get('record')
            if record:
                if isinstance(record, MemoryRecord):
                    examples_text += f"- \"{record.source_column}\" was mapped to \"{record.target_field}\"\n"
                elif isinstance(record, dict):
                    examples_text += f"- \"{record.get('source_column')}\" was mapped to \"{record.get('target_field')}\"\n"
    
    # Construct final prompt
    prompt = f"""You are a schema mapping expert. Given a source column and candidate target fields, select the best match.

Source Column Information:
- Name: {source_profile.name}
- Data Type: {source_profile.inferred_dtype}
- Null Ratio: {source_profile.null_ratio:.2%}
- Uniqueness: {source_profile.uniqueness_ratio:.2%}

Candidate Target Fields:
{candidates_text}
{examples_text}

Task: Select the best target field from the candidates above. Consider:
1. Semantic similarity between source column name and target field name
2. Data type compatibility (if applicable)
3. Sample values and their relationship to the field description
4. Historical mapping patterns (if provided)
5. Whether the field is required

Respond with ONLY valid JSON in this exact format:
{{
  "target": "field_name_of_best_match",
  "explanation": "Brief explanation of why this is the best match",
  "confidence": 0.95
}}

The confidence should be between 0.0 and 1.0, where:
- 0.9-1.0: Very confident, clear semantic match
- 0.7-0.9: Confident, good match with minor ambiguity
- 0.5-0.7: Moderate confidence, reasonable match but alternatives exist
- Below 0.5: Low confidence, unclear which is best
"""
    
    return prompt


def parse_llm_response(response: Dict) -> Optional[Dict]:
    """
    Extract selected target field and explanation from LLM JSON response.
    
    Parses the OpenRouter/OpenAI-compatible response and extracts the structured decision.

    Args:
        response: Raw response dictionary from chat completions API
    
    Returns:
        Dictionary with 'target', 'explanation', and 'confidence' keys, or None if parsing fails
    """
    if not response:
        return None
    
    try:
        # Extract message content from OpenAI-compatible response format
        choices = response.get('choices', [])
        if not choices:
            print("Error: No choices in LLM response")
            return None
        
        message = choices[0].get('message', {})
        content = message.get('content', '')
        
        if not content:
            print("Error: Empty content in LLM response")
            return None
        
        # Parse JSON from content
        # Try to extract JSON if wrapped in markdown code blocks
        content = content.strip()
        if content.startswith('```'):
            # Remove markdown code blocks
            lines = content.split('\n')
            content = '\n'.join(lines[1:-1]) if len(lines) > 2 else content
        
        result = json.loads(content)
        
        # Validate expected keys
        if 'target' not in result:
            print("Error: 'target' key missing in LLM response")
            return None
        
        # Ensure confidence is present and valid
        if 'confidence' not in result:
            result['confidence'] = 0.5  # Default moderate confidence
        
        result['confidence'] = float(result['confidence'])
        result['confidence'] = max(0.0, min(1.0, result['confidence']))  # Clamp to [0, 1]
        
        if 'explanation' not in result:
            result['explanation'] = "No explanation provided"
        
        return result
    
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse LLM response as JSON: {e}")
        print(f"Content: {content}")
        return None
    
    except (KeyError, ValueError) as e:
        print(f"Error: Invalid LLM response structure: {e}")
        return None


def calculate_llm_confidence(llm_result: Dict, candidates: List[Dict], 
                            heuristic_top: Optional[Dict] = None) -> float:
    """
    Derive confidence from margin, consistency, and agreement with heuristics.
    
    Calculates a final confidence score by considering:
    - LLM's own confidence score
    - Agreement with top heuristic candidate
    - Score margin between candidates
    
    Args:
        llm_result: Parsed LLM response with 'target', 'confidence', etc.
        candidates: List of candidate dicts with scores
        heuristic_top: Optional top heuristic candidate for comparison
    
    Returns:
        Final confidence score (0.0 to 1.0)
    """
    llm_confidence = llm_result.get('confidence', 0.0)
    llm_target = llm_result.get('target')
    
    # Start with LLM's confidence
    final_confidence = llm_confidence
    
    # Check if LLM agrees with top heuristic candidate
    if heuristic_top:
        heuristic_target_name = None
        if hasattr(heuristic_top, 'field_name'):
            heuristic_target_name = heuristic_top.field_name
        elif isinstance(heuristic_top, dict):
            target = heuristic_top.get('target')
            if hasattr(target, 'field_name'):
                heuristic_target_name = target.field_name
            elif isinstance(target, dict):
                heuristic_target_name = target.get('field_name')
        
        if heuristic_target_name and llm_target == heuristic_target_name:
            # LLM agrees with heuristics - boost confidence
            final_confidence = min(final_confidence + 0.15, 1.0)
    
    # Calculate score margin if multiple candidates
    if len(candidates) >= 2:
        scores = [c.get('score', 0.0) for c in candidates]
        if scores[0] > 0:
            margin = (scores[0] - scores[1]) / scores[0]
            
            # Large margin = more confident
            if margin > 0.3:
                final_confidence = min(final_confidence + 0.05, 1.0)
            elif margin < 0.1:
                # Small margin = less confident
                final_confidence = max(final_confidence - 0.05, 0.0)
    
    return final_confidence


def get_rate_limiter_stats() -> Dict[str, Any]:
    """
    Get current rate limiter statistics.
    
    Returns:
        Dictionary with rate limiter stats
    """
    return _rate_limiter.get_stats()


def get_cache_stats() -> Dict[str, Any]:
    """
    Get current cache statistics.
    
    Returns:
        Dictionary with cache stats
    """
    return _response_cache.get_stats()


def clear_cache():
    """Clear the LLM response cache."""
    _response_cache.clear()
    print("LLM response cache cleared")


def llm_select(source_profile: Any, candidates: List[Dict], 
               memory_examples: List[Dict] = None) -> Dict[str, Optional[str]]:
    """
    Use LLM to select the best target field from top candidates.
    
    This is the main entry point for LLM-based selection. It:
    1. Builds a structured prompt
    2. Calls OpenRouter API
    3. Parses the response
    4. Calculates final confidence
    
    Falls back to top heuristic candidate if LLM is unavailable or fails.
    
    Args:
        source_profile: ColumnProfile or dict with source column details
        candidates: List of candidate dicts with 'target' and 'score' keys
        memory_examples: Optional list of memory search results
    
    Returns:
        Dictionary with 'selected_target', 'explanation', and 'confidence' keys
    """
    # Handle empty candidate list
    if not candidates:
        return {
            "selected_target": None,
            "explanation": "No candidates available.",
            "confidence": 0.0
        }
    
    # Convert dict to ColumnProfile if needed
    if isinstance(source_profile, dict):
        from backend.models import ColumnProfile
        source_profile = ColumnProfile(**source_profile)
    
    # Build prompt
    prompt = build_llm_prompt(source_profile, candidates, memory_examples)
    
    # Call OpenRouter API
    api_response = call_llm_api(prompt)
    
    # If API fails, fall back to top heuristic candidate
    if not api_response:
        top_candidate = candidates[0]
        target = top_candidate.get('target')
        
        target_name = None
        if hasattr(target, 'field_name'):
            target_name = target.field_name
        elif isinstance(target, dict):
            target_name = target.get('field_name', 'Unknown')

        print(f"LLM not responded: Top heuristic candidate selected {target_name}")
        
        return {
            "selected_target": target_name,
            "explanation": "LLM integration unavailable. Using top heuristic candidate.",
            "confidence": top_candidate.get('score', 0.0)
        }
    
    # Parse LLM response
    llm_result = parse_llm_response(api_response)
    
    if not llm_result:
        # Parsing failed, fall back to top candidate
        top_candidate = candidates[0]
        target = top_candidate.get('target')
        
        target_name = None
        if hasattr(target, 'field_name'):
            target_name = target.field_name
        elif isinstance(target, dict):
            target_name = target.get('field_name', 'Unknown')

        print(f"LLM response parsing failed: Heuristic candidate selected {target_name}")
        
        return {
            "selected_target": target_name,
            "explanation": "LLM response parsing failed. Using top heuristic candidate.",
            "confidence": top_candidate.get('score', 0.0)
        }
    
    # Calculate final confidence
    heuristic_top = candidates[0].get('target') if candidates else None
    final_confidence = calculate_llm_confidence(llm_result, candidates, heuristic_top)
    
    return {
        "selected_target": llm_result['target'],
        "explanation": llm_result['explanation'],
        "confidence": final_confidence
    }
