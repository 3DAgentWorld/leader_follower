#!/usr/bin/env python
# encoding: utf-8
"""
ChatGPT API wrapper.
Supports custom api_key and api_base.
"""
import time
import warnings
from typing import List, Optional

import openai
from openai import OpenAI


def chatgpt(model: str, messages: List[dict], temperature: float, 
            api_key: Optional[str] = None, api_base: Optional[str] = None,
            extra_body: Optional[dict] = None) -> str:
    """
    Call the ChatGPT API.
    
    Args:
        model: Model name.
        messages: Message list.
        temperature: Temperature parameter.
        api_key: API key. If None, uses global config.
        api_base: API base URL. If None, uses default OpenAI API.
        extra_body: Extra parameters to pass to the API (e.g., reasoning_effort).
    
    Returns:
        Model output text.
    """
    # Prioritize passed-in parameters, otherwise use global config
    _api_key = api_key or openai.api_key
    _api_base = api_base or getattr(openai, 'base_url', None)
    
    if _api_base:
        client = OpenAI(api_key=_api_key, base_url=_api_base)
    else:
        client = OpenAI(api_key=_api_key)
    
    retry = 0
    max_retry = 10
    flag = False
    out = ''
    
    while retry < max_retry and not flag:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=4096,
                extra_body=extra_body
            )
            out = response.choices[0].message.content
            flag = True
        except openai.APIStatusError as e:
            if e.message == "Error code: 307":
                retry += 1
                warnings.warn(f"{e} retry:{retry}")
                time.sleep(1)
                continue
            else:
                if retry < max_retry:
                    retry += 1
                    warnings.warn(f"{e} retry:{retry}")
                    time.sleep(2)
                    continue
                else:
                    raise e
        except openai.RateLimitError as e:
            retry += 1
            warnings.warn(f"Rate limit error: {e}, retry:{retry}")
            time.sleep(5)
            continue
        except Exception as e:
            if retry < max_retry:
                retry += 1
                warnings.warn(f"{e} retry:{retry}")
                time.sleep(2)
                continue
            else:
                raise e
    
    client.close()
    return out
