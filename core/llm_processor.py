"""
LLM Processor Module

This module provides comprehensive Large Language Model integration with support for
multiple providers (OpenAI, Anthropic, etc.), intelligent prompt engineering,
response processing, and decision-making assistance. It implements advanced features
including context management, token optimization, and provider fallback mechanisms.

Author: GG
Version: 0.1.0
Date: 2025-09-16
"""

import base64
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import anthropic
import httpx
import openai
import tiktoken
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from PIL import Image

from config.logging_config import get_logger, get_performance_logger
from config.settings import get_config
from models.data_models import ProcessingTask
from models.llm_models import LLMRequest, PromptTemplate
from utils.exceptions import (LLMProcessingError, ProviderUnavailableError,
                              RateLimitExceededError)
from utils.helpers import estimate_tokens, retry_with_backoff, truncate_text


class ProviderType(Enum):
    """Enumeration of supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


@dataclass
class ProcessingResult:
    """
    Result of LLM processing operation.

    This class encapsulates the complete result of an LLM processing request
    including the generated response, token usage, performance metrics,
    and any metadata associated with the processing.
    """

    # Response data
    content: str
    reasoning: Optional[str] = None
    confidence: Optional[float] = None

    # Processing metadata
    provider: str = ""
    model: str = ""
    processing_time_ms: float = 0.0

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0

    # Request context
    request_id: Optional[str] = None


    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            'content': self.content,
            'reasoning': self.reasoning,
            'confidence': self.confidence,
            'provider': self.provider,
            'model': self.model,
            'processing_time_ms': self.processing_time_ms,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'total_tokens': self.total_tokens,
            'estimated_cost': self.estimated_cost,
            'request_id': self.request_id,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


class PromptEngine:
    """
    Advanced prompt engineering and template management.

    This class provides sophisticated prompt construction, template management,
    and context optimization for various LLM providers and use cases.
    """

    def __init__(self):
        """Initialize prompt engine with templates and utilities."""
        self.templates = self._load_prompt_templates()
        self.context_limits = {
            'gpt-4': 128000,
            'gpt-4-32k': 32000,
            'gpt-3.5-turbo': 16000,
            'claude-3-opus': 200000,
            'claude-3-sonnet': 200000,
            'claude-3-haiku': 200000
        }

        self.logger = get_logger(__name__)
        self.logger.debug("Prompt engine initialized with templates")

    def _load_prompt_templates(self) -> Dict[str, PromptTemplate]:
        """Load and initialize prompt templates."""
        templates = {}

        # Data Analysis Template
        templates['data_analysis'] = PromptTemplate(
            name='data_analysis',
            system_prompt="""You are an expert data analyst and decision-making assistant. Your role is to:
1. Analyze provided data (text, images, documents, videos)
2. Extract key insights and patterns
3. Provide actionable recommendations
4. Make informed decisions about next steps

Always structure your responses with clear reasoning and confidence levels.""",

            user_prompt="""Analyze the following data and provide insights:

{data_context}

Please provide:
1. **Summary**: Brief overview of the data
2. **Key Insights**: Main findings and patterns
3. **Recommendations**: Suggested actions
4. **Decision**: Recommended next step with reasoning
5. **Confidence**: Rate your confidence (0-1) in this analysis

Response format should be JSON with the following structure:
{
    "summary": "Brief overview",
    "insights": ["insight1", "insight2", ...],
    "recommendations": ["rec1", "rec2", ...],
    "decision": "recommended_action",
    "reasoning": "detailed reasoning",
    "confidence": 0.85
}""",
            variables=['data_context']
        )

        # Decision Making Template
        templates['decision_making'] = PromptTemplate(
            name='decision_making',
            system_prompt="""You are a smart decision-making AI that helps choose between different actions based on analyzed data.

Available actions:
1. CREATE_WOOCOMMERCE_POST - Create a draft product post for WooCommerce
2. UPDATE_EXCEL_DATABASE - Update Excel cloud database with data
3. REQUEST_MORE_DATA - Ask user for additional information
4. CHAIN_LLM_PROCESSING - Use another LLM for further analysis

Consider the data type, content quality, user intent, and business logic when making decisions.""",

            user_prompt="""Based on this analysis, decide what action to take:

**Analysis Results:**
{analysis_results}

**Available Data:**
{available_data}

**User Context:**
{user_context}

Choose the most appropriate action and provide detailed reasoning. Format your response as JSON:
{
    "action": "ACTION_NAME",
    "reasoning": "Detailed explanation of why this action is best",
    "confidence": 0.9,
    "parameters": {
        "param1": "value1",
        "param2": "value2"
    },
    "fallback_action": "ALTERNATIVE_ACTION"
}""",
            variables=['analysis_results', 'available_data', 'user_context']
        )

        # WooCommerce Product Template
        templates['woocommerce_product'] = PromptTemplate(
            name='woocommerce_product',
            system_prompt="""You are an expert e-commerce product listing creator. Generate compelling product descriptions, titles, and metadata for WooCommerce based on provided data (images, descriptions, specifications).""",

            user_prompt="""Create a WooCommerce product listing based on this data:

{product_data}

Generate the following in JSON format:
{
    "title": "Compelling product title",
    "description": "Detailed HTML product description",
    "short_description": "Brief summary",
    "price": "suggested_price",
    "categories": ["category1", "category2"],
    "tags": ["tag1", "tag2"],
    "attributes": {
        "attribute_name": "attribute_value"
    },
    "meta_description": "SEO meta description",
    "sku": "suggested_sku"
}""",
            variables=['product_data']
        )

        # Excel Data Template
        templates['excel_data'] = PromptTemplate(
            name='excel_data',
            system_prompt="""You are a data processing specialist that structures information for Excel databases. Extract and organize data in a format suitable for spreadsheet storage.""",

            user_prompt="""Process this data for Excel database entry:

{input_data}

Structure the data as a JSON object with clear field names and values:
{
    "worksheet": "target_worksheet_name",
    "data": {
        "column_name_1": "value_1",
        "column_name_2": "value_2",
        "timestamp": "2025-09-16T10:30:00Z"
    },
    "metadata": {
        "source": "telegram_bot",
        "processed_at": "timestamp"
    }
}""",
            variables=['input_data']
        )
        
        # Follow-up Questions Template
        templates['followup_questions'] = PromptTemplate(
            name='followup_questions',
            system_prompt="""You are a helpful assistant that asks clarifying questions to gather more information for better decision making.""",
            
            user_prompt="""Based on the current data, generate helpful follow-up questions:

{current_data}

Generate 3-5 specific questions that would help improve the analysis or decision making:
{
    "questions": [
        "What is the specific use case for this product?",
        "What is your target price range?",
        "Do you have additional product specifications?"
    ],
    "reason": "Why these questions are important",
    "priority": "high|medium|low"
}""",
            variables=['current_data']
        )
        
        return templates

    def get_template(self, template_name: str) -> Optional[PromptTemplate]:
        """
        Get prompt template by name.
        
        Args:
            template_name: Name of the template
            
        Returns:
            PromptTemplate object or None if not found
        """
        return self.templates.get(template_name)

    def render_prompt(self, template_name: str, variables: Dict[str, Any]) -> Tuple[str, str]:
        """
        Render prompt template with provided variables.
        
        Args:
            template_name: Name of the template to render
            variables: Dictionary of template variables
            
        Returns:
            Tuple of (system_prompt, user_prompt)
            
        Raises:
            ValueError: If template not found or variables missing
        """
        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        try:
            # Check for missing variables
            missing_vars = set(template.variables) - set(variables.keys())
            if missing_vars:
                raise ValueError(f"Missing template variables: {missing_vars}")
            
            # Render prompts
            system_prompt = template.system_prompt.format(**variables) if template.system_prompt else ""
            user_prompt = template.user_prompt.format(**variables)
            
            self.logger.debug(f"Rendered template '{template_name}' with {len(variables)} variables")
            
            return system_prompt, user_prompt
            
        except KeyError as e:
            raise ValueError(f"Template variable error: {e}")
        except Exception as e:
            raise ValueError(f"Template rendering error: {e}")

    def optimize_for_context(self, text: str, model: str, max_tokens: int = None) -> str:
        """
        Optimize text content for model context limits.
        
        Args:
            text: Text content to optimize
            model: Model name for context limits
            max_tokens: Optional token limit override
            
        Returns:
            Optimized text content
        """
        try:
            # Get context limit for model
            context_limit = max_tokens or self.context_limits.get(model, 4000)
            
            # Estimate current token count
            current_tokens = estimate_tokens(text, model)
            
            if current_tokens <= context_limit * 0.8:  # Use 80% of limit for safety
                return text
            
            # Truncate if needed
            target_tokens = int(context_limit * 0.7)  # Leave room for response
            optimized_text = truncate_text(text, target_tokens)
            
            self.logger.debug(f"Optimized text from {current_tokens} to {estimate_tokens(optimized_text, model)} tokens")
            
            return optimized_text
            
        except Exception as e:
            self.logger.warning(f"Context optimization failed: {e}, using original text")
            return text


class OpenAIProvider:
    """
    OpenAI API provider implementation.

    This class handles all interactions with OpenAI's API including
    chat completions, vision processing, and token management.
    """

    def __init__(self, api_key: str):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.provider_type = ProviderType.OPENAI
        self.models = {
            'gpt-4': {
                'max_tokens': 8192,
                'supports_vision': True,
                'cost_per_1k_tokens': {'input': 0.03, 'output': 0.06}
            },
            'gpt-4-turbo': {
                'max_tokens': 4096,
                'supports_vision': True,
                'cost_per_1k_tokens': {'input': 0.01, 'output': 0.03}
            },
            'gpt-3.5-turbo': {
                'max_tokens': 4096,
                'supports_vision': False,
                'cost_per_1k_tokens': {'input': 0.0015, 'output': 0.002}
            }
        }

        self.logger = get_logger(__name__)
        self.logger.debug("OpenAI provider initialized")

    async def process_request(self, request: LLMRequest) -> ProcessingResult:
        """
        Process LLM request using OpenAI API.

        Args:
            request: LLM request object

        Returns:
            Processing result

        Raises:
            LLMProcessingError: If processing fails
        """
        start_time = time.time()

        try:
            # Prepare messages
            messages = self._prepare_messages(request)

            # Make API request
            response = await self.client.chat.completions.create(
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                top_p=request.top_p,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty
            )

            # Process response
            result = self._process_response(response, request, start_time)

            self.logger.info(f"OpenAI request completed in {result.processing_time_ms:.1f}ms")

            return result

        except openai.RateLimitError as e:
            self.logger.error(f"OpenAI rate limit exceeded: {e}")
            raise RateLimitExceededError(f"OpenAI rate limit: {e}")
        except openai.AuthenticationError as e:
            self.logger.error(f"OpenAI authentication error: {e}")
            raise ProviderUnavailableError(f"OpenAI auth error: {e}")
        except openai.APIError as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise LLMProcessingError(f"OpenAI API error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected OpenAI error: {e}")
            raise LLMProcessingError(f"OpenAI processing failed: {e}")
    
    def _prepare_messages(self, request: LLMRequest) -> List[Dict[str, Any]]:
        """
        Prepare messages for OpenAI API format.
        
        Args:
            request: LLM request object
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # Add system message if provided
        if request.system_prompt:
            messages.append({
                "role": "system",
                "content": request.system_prompt
            })
        
        # Add user message
        if request.has_images and self.models[request.model]['supports_vision']:
            # Vision request with images
            content = [{"type": "text", "text": request.user_prompt}]
            
            for image_data in request.image_data:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}",
                        "detail": "high"
                    }
                })
            
            messages.append({
                "role": "user",
                "content": content
            })
        else:
            # Text-only request
            messages.append({
                "role": "user",
                "content": request.user_prompt
            })
        
        return messages
    
    def _process_response(self, response, request: LLMRequest, start_time: float) -> ProcessingResult:
        """
        Process OpenAI API response into result object.
        
        Args:
            response: OpenAI API response
            request: Original request
            start_time: Request start time
            
        Returns:
            ProcessingResult object
        """
        processing_time = (time.time() - start_time) * 1000
        
        # Extract response content
        content = response.choices[0].message.content
        
        # Extract token usage
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens
        
        # Calculate cost
        model_info = self.models.get(request.model, {})
        cost_info = model_info.get('cost_per_1k_tokens', {'input': 0, 'output': 0})
        
        estimated_cost = (
            (input_tokens / 1000) * cost_info['input'] +
            (output_tokens / 1000) * cost_info['output']
        )
        
        return ProcessingResult(
            content=content,
            provider=self.provider_type.value,
            model=request.model,
            processing_time_ms=processing_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=estimated_cost,
            request_id=request.request_id,
            metadata={
                'finish_reason': response.choices[0].finish_reason,
                'response_id': response.id
            }
        )


class AnthropicProvider:
    """
    Anthropic Claude API provider implementation.
    
    This class handles interactions with Anthropic's Claude API
    including message processing and context management.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key
        """
        self.client = AsyncAnthropic(api_key=api_key)
        self.provider_type = ProviderType.ANTHROPIC
        self.models = {
            'claude-3-opus-20240229': {
                'max_tokens': 4096,
                'supports_vision': True,
                'cost_per_1k_tokens': {'input': 0.015, 'output': 0.075}
            },
            'claude-3-sonnet-20240229': {
                'max_tokens': 4096,
                'supports_vision': True,
                'cost_per_1k_tokens': {'input': 0.003, 'output': 0.015}
            },
            'claude-3-haiku-20240307': {
                'max_tokens': 4096,
                'supports_vision': True,
                'cost_per_1k_tokens': {'input': 0.00025, 'output': 0.00125}
            }
        }
        
        self.logger.debug("Anthropic provider initialized")
    
    async def process_request(self, request: LLMRequest) -> ProcessingResult:
        """
        Process LLM request using Anthropic API.
        
        Args:
            request: LLM request object
            
        Returns:
            Processing result
            
        Raises:
            LLMProcessingError: If processing fails
        """
        start_time = time.time()
        
        try:
            # Prepare messages
            messages = self._prepare_messages(request)
            
            # Make API request
            response = await self.client.messages.create(
                model=request.model,
                messages=messages,
                system=request.system_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            )
            
            # Process response
            result = self._process_response(response, request, start_time)
            
            self.logger.info(f"Anthropic request completed in {result.processing_time_ms:.1f}ms")
            
            return result
            
        except anthropic.RateLimitError as e:
            self.logger.error(f"Anthropic rate limit exceeded: {e}")
            raise RateLimitExceededError(f"Anthropic rate limit: {e}")
        except anthropic.AuthenticationError as e:
            self.logger.error(f"Anthropic authentication error: {e}")
            raise ProviderUnavailableError(f"Anthropic auth error: {e}")
        except anthropic.APIError as e:
            self.logger.error(f"Anthropic API error: {e}")
            raise LLMProcessingError(f"Anthropic API error: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected Anthropic error: {e}")
            raise LLMProcessingError(f"Anthropic processing failed: {e}")
    
    def _prepare_messages(self, request: LLMRequest) -> List[Dict[str, Any]]:
        """Prepare messages for Anthropic API format."""
        messages = []
        
        if request.has_images and self.models[request.model]['supports_vision']:
            # Vision request with images
            content = [{"type": "text", "text": request.user_prompt}]
            
            for image_data in request.image_data:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data
                    }
                })
            
            messages.append({
                "role": "user", 
                "content": content
            })
        else:
            # Text-only request
            messages.append({
                "role": "user",
                "content": request.user_prompt
            })
        
        return messages
    
    def _process_response(self, response, request: LLMRequest, start_time: float) -> ProcessingResult:
        """Process Anthropic API response into result object."""
        processing_time = (time.time() - start_time) * 1000
        
        # Extract response content
        content = response.content[0].text
        
        # Extract token usage
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = input_tokens + output_tokens
        
        # Calculate cost
        model_info = self.models.get(request.model, {})
        cost_info = model_info.get('cost_per_1k_tokens', {'input': 0, 'output': 0})
        
        estimated_cost = (
            (input_tokens / 1000) * cost_info['input'] +
            (output_tokens / 1000) * cost_info['output']
        )
        
        return ProcessingResult(
            content=content,
            provider=self.provider_type.value,
            model=request.model,
            processing_time_ms=processing_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=estimated_cost,
            request_id=request.request_id,
            metadata={
                'stop_reason': response.stop_reason,
                'response_id': response.id
            }
        )


class LLMProcessor:
    """
    Main LLM processor with multi-provider support and intelligent routing.

    This class provides the primary interface for LLM processing with
    advanced features including provider fallback, response validation,
    context management, and cost optimization.
    """

    def __init__(self):
        """Initialize LLM processor with configuration and providers."""
        self.config = get_config()

        self.prompt_engine = PromptEngine()
        self.providers: Dict[str, Union[OpenAIProvider, AnthropicProvider]] = {}
        self.usage_stats: Dict[str, Dict[str, Any]] = {}
        self.request_history: List[Dict[str, Any]] = []

        # Configure module logger
        self.logger = get_logger(__name__)
        self.perf_logger = get_performance_logger(__name__)
        self.logger.info("LLM processor initialized")

        # Initialize providers
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize available LLM providers."""
        try:
            # Initialize OpenAI provider
            if self.config.llm.openai_api_key:
                self.providers[ProviderType.OPENAI.value] = OpenAIProvider(
                    self.config.llm.openai_api_key
                )
                self.logger.debug("OpenAI provider initialized")

            # Initialize Anthropic provider
            if self.config.llm.anthropic_api_key:
                self.providers[ProviderType.ANTHROPIC.value] = AnthropicProvider(
                    self.config.llm.anthropic_api_key
                )
                self.logger.debug("Anthropic provider initialized")

            if not self.providers:
                self.logger.warning("No LLM providers configured")

        except Exception as e:
            self.logger.error(f"Error initializing LLM providers: {e}")
            raise

    async def analyze_data(self, task: ProcessingTask) -> ProcessingResult:
        """
        Analyze data using appropriate LLM provider.
        
        Args:
            task: Processing task containing data to analyze
            
        Returns:
            Analysis results
            
        Raises:
            LLMProcessingError: If analysis fails
        """
        decorator = self.perf_logger.log_function_performance("analyze_data")
        
        @decorator
        async def _analyze_data_inner():
            try:
                # Prepare data context
                data_context = await self._prepare_data_context(task)
                
                # Create LLM request
                request = await self._create_analysis_request(data_context, task)
                
                # Process with primary provider
                try:
                    result = await self._process_with_provider(
                        request, 
                        self.config.llm.primary_provider
                    )
                    
                    # Validate and parse result
                    parsed_result = await self._parse_analysis_result(result.content)
                    result.metadata.update(parsed_result)
                    
                    return result
                    
                except (ProviderUnavailableError, RateLimitExceededError) as e:
                    self.logger.warning(f"Primary provider failed: {e}, trying fallback")
                    
                    # Try fallback providers
                    for fallback_provider in self.config.llm.fallback_providers:
                        try:
                            result = await self._process_with_provider(request, fallback_provider)
                            parsed_result = await self._parse_analysis_result(result.content)
                            result.metadata.update(parsed_result)
                            return result
                        except Exception as fallback_error:
                            self.logger.warning(f"Fallback provider {fallback_provider} failed: {fallback_error}")
                            continue
                    
                    # All providers failed
                    raise LLMProcessingError("All LLM providers failed")
            
            except Exception as e:
                self.logger.error(f"Data analysis failed: {e}")
                raise LLMProcessingError(f"Analysis failed: {e}")

        await _analyze_data_inner()
    
    async def make_decision(self, analysis_result: ProcessingResult, 
                           task: ProcessingTask) -> ProcessingResult:
        """
        Make decision based on analysis results.
        
        Args:
            analysis_result: Results from data analysis
            task: Original processing task

        Returns:
            Decision results
        """
        decorator = self.perf_logger.log_function_performance("make_decision")

        @decorator
        async def _make_decision_inner():
            try:
                # Prepare decision context
                decision_context = await self._prepare_decision_context(
                    analysis_result, task
                )

                # Create decision request
                request = await self._create_decision_request(decision_context)

                # Process decision
                result = await self._process_with_provider(
                    request,
                    self.config.llm.primary_provider
                )

                # Validate and parse decision
                decision_data = await self._parse_decision_result(result.content)
                result.metadata.update(decision_data)

                return result

            except Exception as e:
                self.logger.error(f"Decision making failed: {e}")
                raise LLMProcessingError(f"Decision failed: {e}")

        await _make_decision_inner()

    async def generate_woocommerce_content(self, data: Dict[str, Any]) -> ProcessingResult:
        """
        Generate WooCommerce product content.

        Args:
            data: Product data

        Returns:
            Generated content
        """
        try:
            # Prepare WooCommerce request
            system_prompt, user_prompt = self.prompt_engine.render_prompt(
                'woocommerce_product',
                {'product_data': json.dumps(data, indent=2)}
            )

            request = LLMRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.config.llm.openai_model,
                temperature=0.7,
                max_tokens=2000
            )

            # Process request
            result = await self._process_with_provider(
                request, 
                self.config.llm.primary_provider
            )

            # Parse product data
            product_data = await self._parse_json_response(result.content)
            result.metadata.update(product_data)

            return result

        except Exception as e:
            self.logger.error(f"WooCommerce content generation failed: {e}")
            raise LLMProcessingError(f"Content generation failed: {e}")

    async def generate_excel_data(self, data: Dict[str, Any]) -> ProcessingResult:
        """
        Generate Excel-formatted data.

        Args:
            data: Input data

        Returns:
            Structured Excel data
        """
        try:
            # Prepare Excel request
            system_prompt, user_prompt = self.prompt_engine.render_prompt(
                'excel_data',
                {'input_data': json.dumps(data, indent=2)}
            )

            request = LLMRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.config.llm.openai_model,
                temperature=0.3,  # Lower temperature for structured data
                max_tokens=1500
            )

            # Process request
            result = await self._process_with_provider(
                request,
                self.config.llm.primary_provider
            )

            # Parse Excel data
            excel_data = await self._parse_json_response(result.content)
            result.metadata.update(excel_data)

            return result

        except Exception as e:
            self.logger.error(f"Excel data generation failed: {e}")
            raise LLMProcessingError(f"Excel generation failed: {e}")
    
    async def generate_followup_questions(self, context: Dict[str, Any]) -> ProcessingResult:
        """
        Generate follow-up questions for more data.
        
        Args:
            context: Current context data
            
        Returns:
            Generated questions
        """
        try:
            # Prepare questions request
            system_prompt, user_prompt = self.prompt_engine.render_prompt(
                'followup_questions',
                {'current_data': json.dumps(context, indent=2)}
            )
            
            request = LLMRequest(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.config.llm.openai_model,
                temperature=0.8,  # Higher temperature for creative questions
                max_tokens=1000
            )
            
            # Process request
            result = await self._process_with_provider(
                request,
                self.config.llm.primary_provider
            )
            
            # Parse questions
            questions_data = await self._parse_json_response(result.content)
            result.metadata.update(questions_data)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Follow-up questions generation failed: {e}")
            raise LLMProcessingError(f"Questions generation failed: {e}")
    
    async def _prepare_data_context(self, task: ProcessingTask) -> str:
        """
        Prepare data context for analysis.
        
        Args:
            task: Processing task
            
        Returns:
            Formatted data context string
        """
        context_parts = []
        
        # Add text data
        if 'text' in task.data:
            context_parts.append(f"**Text Data:**\n{task.data['text']}\n")
        
        # Add media file information
        if 'media_files' in task.data:
            media_info = []
            for media in task.data['media_files']:
                info = f"- {media.get('file_type', 'unknown')}: {media.get('file_name', 'unnamed')}"
                if media.get('caption'):
                    info += f" (Caption: {media['caption']})"
                media_info.append(info)
            
            if media_info:
                context_parts.append(f"**Media Files:**\n" + "\n".join(media_info) + "\n")
        
        # Add session context
        if 'session_data' in task.data:
            session_data = task.data['session_data']
            context_parts.append(f"**Session Info:**\n- Files: {len(session_data.get('collected_media', []))}")
            context_parts.append(f"- Messages: {len(session_data.get('conversation_history', []))}\n")
        
        return "\n".join(context_parts)
    
    async def _create_analysis_request(self, data_context: str, task: ProcessingTask) -> LLMRequest:
        """Create LLM request for data analysis."""
        # Check if task has images for vision processing
        has_images = False
        image_data = []
        
        if 'media_files' in task.data:
            for media in task.data['media_files']:
                if media.get('file_type') == 'photo' and media.get('local_path'):
                    try:
                        # Load and encode image
                        image_path = Path(media['local_path'])
                        if image_path.exists():
                            with open(image_path, 'rb') as img_file:
                                encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
                                image_data.append(encoded_image)
                                has_images = True
                    except Exception as e:
                        self.logger.warning(f"Failed to load image {media.get('local_path')}: {e}")
        
        # Render prompt template
        system_prompt, user_prompt = self.prompt_engine.render_prompt(
            'data_analysis',
            {'data_context': data_context}
        )
        
        return LLMRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.config.llm.openai_model,
            temperature=self.config.llm.openai_temperature,
            max_tokens=self.config.llm.openai_max_tokens,
            has_images=has_images,
            image_data=image_data,
            request_id=task.id
        )
    
    async def _prepare_decision_context(self, analysis: ProcessingResult, 
                                      task: ProcessingTask) -> Dict[str, Any]:
        """Prepare context for decision making."""
        return {
            'analysis_results': analysis.metadata,
            'available_data': {
                'media_files': len(task.data.get('media_files', [])),
                'text_data': bool(task.data.get('text')),
                'session_active': bool(task.data.get('session_data'))
            },
            'user_context': {
                'user_id': task.user_id,
                'task_type': task.task_type,
                'priority': task.priority.value if task.priority else 'normal'
            }
        }
    
    async def _create_decision_request(self, context: Dict[str, Any]) -> LLMRequest:
        """Create LLM request for decision making."""
        system_prompt, user_prompt = self.prompt_engine.render_prompt(
            'decision_making',
            {
                'analysis_results': json.dumps(context['analysis_results'], indent=2),
                'available_data': json.dumps(context['available_data'], indent=2),
                'user_context': json.dumps(context['user_context'], indent=2)
            }
        )
        
        return LLMRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.config.llm.openai_model,
            temperature=0.5,  # Lower temperature for decision making
            max_tokens=1500
        )
    
    async def _process_with_provider(self, request: LLMRequest, 
                                   provider_name: str) -> ProcessingResult:
        """
        Process request with specified provider.
        
        Args:
            request: LLM request
            provider_name: Provider to use
            
        Returns:
            Processing result
            
        Raises:
            ProviderUnavailableError: If provider not available
        """
        provider = self.providers.get(provider_name)
        if not provider:
            raise ProviderUnavailableError(f"Provider {provider_name} not available")
        
        # Apply retry logic for transient failures
        return await retry_with_backoff(
            provider.process_request,
            request,
            max_retries=self.config.llm.max_retries,
            delay=self.config.llm.retry_delay_seconds
        )
    
    async def _parse_json_response(self, response_content: str) -> Dict[str, Any]:
        """
        Parse JSON response from LLM.
        
        Args:
            response_content: Response content string
            
        Returns:
            Parsed JSON data
        """
        try:
            # Try to find JSON in response
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_content[start_idx:end_idx + 1]
                return json.loads(json_str)
            
            # Fallback: try parsing entire response
            return json.loads(response_content)
            
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON response: {e}")
            return {
                'raw_response': response_content,
                'parse_error': str(e)
            }
    
    async def _parse_analysis_result(self, content: str) -> Dict[str, Any]:
        """Parse analysis result from LLM response."""
        return await self._parse_json_response(content)
    
    async def _parse_decision_result(self, content: str) -> Dict[str, Any]:
        """Parse decision result from LLM response."""
        return await self._parse_json_response(content)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all providers."""
        return self.usage_stats
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return list(self.providers.keys())


# Export main classes
__all__ = [
    'LLMProcessor',
    'ProcessingResult', 
    'PromptEngine',
    'OpenAIProvider',
    'AnthropicProvider',
    'ProviderType'
]
