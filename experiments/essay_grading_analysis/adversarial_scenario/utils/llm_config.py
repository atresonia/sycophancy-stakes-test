from concurrent.futures import ThreadPoolExecutor
import os
from typing import Any, Dict, List
from abc import ABC, abstractmethod
from openai import OpenAI
from google import genai
from google.genai import types as genai_types
from anthropic import Anthropic

MAX_WORKERS = 8


class BaseLLM(ABC):
    @abstractmethod
    def get_name(self):
        raise NotImplementedError("get_name() must be implemented in a subclass")

    @abstractmethod
    def generate(self, query: str, sampling_params: Dict[str, Any] = None):
        raise NotImplementedError("generate() must be implemented in a subclass")
    
    def batch_generate(self, queries: List[str], sampling_params: Dict[str, Any] = None) -> List[str]:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            return list(executor.map(lambda q: self.generate(q, sampling_params), queries))


class OpenAILLM(BaseLLM):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_name(self):
        return self.model_name

    def generate(self, query: str, sampling_params: Dict[str, Any] = None):
        sp = sampling_params or {}
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": query}],
            temperature=sp.get("temperature", 1.0),
            max_tokens=sp.get("max_tokens", 1024),
        )
        return response.choices[0].message.content
    
    def generate_format(self, query: str, response_format: Any):
        response = self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[{"role": "user", "content": query}],
            response_format=response_format,
        )
        return response.choices[0].message.parsed


class GeminiLLM(BaseLLM):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def get_name(self):
        return self.model_name

    def generate(self, query: str, sampling_params: Dict[str, Any] = None):
        sp = sampling_params or {}

        # add safety setting to not block
        cfg = genai_types.GenerateContentConfig(
            temperature=sp.get("temperature", 1.0),
            max_output_tokens=sp.get("max_tokens", 512),
            thinking_config=genai_types.ThinkingConfig(thinking_budget=128),
            safety_settings=[
                genai_types.SafetySetting(category=c, threshold="BLOCK_NONE")
                for c in ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                      "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")
            ],
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=query,
            config=cfg,
        )

        if response.text:
            return response.text
        cand = response.candidates[0] if response.candidates else None
        fr = getattr(cand, "finish_reason", None)
        parts = getattr(getattr(cand, "content", None), "parts", []) or []
        txt = "".join(getattr(p, "text", "") or "" for p in parts)
        if not txt:
            print(f"No grader text | finish_reason={fr} | feedback={getattr(response,'prompt_feedback',None)}")
        return txt
        

    def generate_format(self, query: str, response_format: Any):
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=query
        )
        return response.text
    

class AnthropicLLM(BaseLLM):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    def get_name(self):
        return self.model_name

    def generate(self, query: str, sampling_params: Dict[str, Any] = None):
        sp = sampling_params or {}
        response = self.client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": query}],
            temperature=sp.get("temperature", 1.0),
            max_tokens=sp.get("max_tokens", 1024),
        )
        return response.content[0].text


class LLMSwitcher:
    """Factory class for creating and managing different LLM implementations"""

    def __init__(self, provider: str, model_name: str):
        """
        Initialize LLM switcher with configuration

        Args:
            config: Configuration for the Language Model
        """
        self.provider = provider
        self.model_name = model_name
        self._llm = self._create_llm()

    def _create_llm(self):
        """
        Create the appropriate LLM based on configuration

        Returns:
            Instantiated Language Model

        Raises:
            ValueError: If an unsupported LLM type is specified
        """
        if self.model_name == "openai":
            return OpenAILLM(model_name=self.model_name)
        elif self.provider == "gemini":
            return GeminiLLM(model_name=self.model_name)
        elif self.provider == "anthropic":
            return AnthropicLLM(model_name=self.model_name)
        else:
            raise ValueError(f"Unsupported LLM type: {self.model_name}")

    def generate(self, query: str, sampling_params: Dict[str, Any] = None) -> str:
        """
        Generate a response using the selected LLM

        Args:
            query: Input query to generate response for
            sampling_params: Optional sampling parameters

        Returns:
            Generated response
        """
        if sampling_params is None:
            sampling_params = {}
        return self._llm.generate(query, sampling_params)

    def batch_generate(
        self, queries: List[str], sampling_params: Dict[str, Any] = None
    ) -> List[str]:
        """
        Generate responses for a batch of queries using the selected LLM

        Args:
            queries: List of input queries
            sampling_params: Optional sampling parameters

        Returns:
            List of generated responses
        """
        if sampling_params is None:
            sampling_params = {}
        return self._llm.batch_generate(queries)
    

    def generate_format(
        self,
        query: str,
        response_format: Any,
        sampling_params: Dict[str, Any] = None,
    ) -> Any:
        if sampling_params is None:
            sampling_params = {}
        return self._llm.generate_format(query, response_format, sampling_params)


    def __repr__(self) -> str:
        return f"LLMSwitcher(type={self._type}, model_kwargs={self.model_kwargs})"