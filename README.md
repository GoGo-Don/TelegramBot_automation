# Telegram LLM Decision Engine

A sophisticated Python application that receives multimedia data from Telegram, processes it through Large Language Models (LLMs), and intelligently decides between multiple actions including WooCommerce posting, Excel database updates, Telegram responses, or cascaded LLM processing.

## 🚀 Features

### Core Capabilities
- **Multi-Modal Data Processing**: Handles images, videos, documents, and text from Telegram
- **Advanced LLM Integration**: Supports OpenAI GPT-4, Anthropic Claude, with intelligent provider fallback
- **Intelligent Decision Making**: AI-powered decision engine with confidence scoring
- **Multi-Action Execution**: Automated workflows for e-commerce, data management, and communication

### Supported Actions
1. **WooCommerce Integration**: Create draft product posts with images, descriptions, and metadata
2. **Excel Database Updates**: Sync data to Microsoft Excel Online via Graph API
3. **Intelligent Responses**: Generate contextual Telegram replies for data clarification
4. **LLM Chaining**: Multi-stage processing with different models for complex analysis

### Advanced Features
- **State Management**: Global and local state tracking with Redis caching
- **Multi-Level Logging**: Comprehensive logging with performance monitoring and structured output
- **Error Handling**: Graceful degradation with detailed error reporting and recovery
- **Media Processing**: Album support, file validation, and intelligent media grouping
- **Rate Limiting**: Built-in protection against API abuse and quota management
- **Health Monitoring**: Real-time system health checks and metrics collection

## 🏗️ Architecture

### Project Structure
```
telegram_llm_decision_engine/
├── main.py                     # Application entry point
├── requirements.txt            # Python dependencies
├── .env.example               # Environment configuration template
├── config/                    # Configuration management
│   ├── settings.py           # Application settings
│   └── logging_config.py     # Logging configuration
├── core/                     # Core business logic
│   ├── telegram_handler.py  # Telegram bot functionality
│   ├── llm_processor.py     # LLM integration and processing
│   ├── decision_engine.py   # Decision-making logic
│   ├── state_manager.py     # State management
│   └── base_handler.py      # Abstract base classes
├── integrations/            # External service integrations
│   ├── woocommerce_handler.py
│   ├── excel_handler.py
│   └── telegram_responder.py
├── models/                  # Data models and schemas
│   ├── data_models.py
│   ├── telegram_models.py
│   ├── llm_models.py
│   └── decision_models.py
├── utils/                   # Utility functions
│   ├── file_manager.py
│   ├── validators.py
│   ├── helpers.py
│   └── exceptions.py
├── storage/                 # Storage and caching
│   ├── local_storage.py
│   ├── cache_manager.py
│   └── session_store.py
├── tests/                   # Test suite
└── docs/                    # Documentation
```

### Technology Stack
- **Python 3.9+**: Modern async/await patterns
- **python-telegram-bot**: Telegram Bot API wrapper
- **OpenAI & Anthropic**: LLM providers
- **SQLAlchemy**: Database ORM
- **Redis**: Caching and session storage
- **Pydantic**: Data validation and settings
- **Structlog**: Structured logging
- **Docker**: Containerization support

## 📋 Prerequisites

### Required
- Python 3.9 or higher
- Telegram Bot Token (from [@BotFather](https://t.me/BotFather))
- At least one LLM API key (OpenAI or Anthropic)

### Optional (for full functionality)
- Redis server (for caching and sessions)
- PostgreSQL (for persistent storage)
- WooCommerce store with REST API access
- Microsoft Azure app registration (for Excel integration)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd telegram_llm_decision_engine
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

### 5. Initialize Database
```bash
python -c "from core.state_manager import StateManager; import asyncio; asyncio.run(StateManager().initialize())"
```

## ⚙️ Configuration

### Essential Settings
Create your `.env` file with these minimum required settings:

```env
# Telegram Bot (Required)
TELEGRAM_BOT_TOKEN="your_bot_token_from_botfather"

# LLM Provider (Choose one or both)
LLM_OPENAI_API_KEY="your_openai_api_key"
LLM_ANTHROPIC_API_KEY="your_anthropic_api_key"
```

### WooCommerce Integration (Optional)
```env
WOOCOMMERCE_STORE_URL="https://your-store.com"
WOOCOMMERCE_CONSUMER_KEY="ck_your_consumer_key"
WOOCOMMERCE_CONSUMER_SECRET="cs_your_consumer_secret"
```

### Excel Integration (Optional)
```env
EXCEL_CLIENT_ID="your-azure-app-client-id"
EXCEL_CLIENT_SECRET="your-azure-app-client-secret"
EXCEL_TENANT_ID="your-azure-tenant-id"
EXCEL_WORKBOOK_ID="your-excel-workbook-id"
```

### Advanced Configuration
The system supports extensive configuration options. See `.env.example` for all available settings including:
- Logging levels and formats
- Rate limiting and security settings
- Performance tuning parameters
- Database and caching options

## 🚀 Usage

### Starting the Application

#### Development Mode
```bash
python main.py dev
```

#### Production Mode
```bash
python main.py prod
```

#### Using Docker
```bash
docker build -t telegram-llm-engine .
docker run -d --env-file .env telegram-llm-engine
```

### Telegram Bot Commands

- `/start` - Initialize bot and get welcome message
- `/collect` - Start interactive data collection
- `/status` - Check current processing status
- `/help` - Display help information
- `/cancel` - Cancel current operation
- `/clear` - Clear session data
- `/settings` - View current bot settings

### Basic Workflow

1. **Start Conversation**: Send `/start` to the bot
2. **Send Data**: Upload images, videos, documents, or send text messages
3. **Processing**: The system analyzes your data using LLMs
4. **Decision Making**: AI determines the best action based on content
5. **Execution**: Automatically performs the chosen action:
   - Creates WooCommerce product draft
   - Updates Excel database
   - Asks for more information
   - Processes with additional LLMs

### Example Use Cases

#### E-commerce Product Creation
1. Send product photos to the bot
2. Add product description as text
3. Bot analyzes images and text
4. Automatically creates WooCommerce product draft
5. Generates SEO-optimized title, description, and metadata

#### Data Collection and Analysis
1. Send multiple files (documents, images, data files)
2. Bot processes and structures the information
3. Updates Excel spreadsheet with organized data
4. Provides analysis summary via Telegram

#### Content Processing Pipeline
1. Submit content for analysis
2. Bot uses multiple LLMs for different aspects
3. Generates comprehensive insights
4. Delivers formatted results through chosen channel

## 🔧 API Reference

### Core Classes

#### `TelegramHandler`
Manages Telegram bot interactions and multimedia processing.

```python
handler = TelegramHandler(config, state_manager)
await handler.initialize()
await handler.start_polling()
```

#### `LLMProcessor`
Handles LLM requests with provider fallback.

```python
processor = LLMProcessor()
result = await processor.analyze_data(task)
decision = await processor.make_decision(result, task)
```

#### `DecisionEngine`
Makes intelligent decisions based on processed data.

```python
engine = DecisionEngine(llm_processor, state_manager)
action = await engine.decide_action(analysis_result)
await engine.execute_action(action)
```

### Data Models

#### `ProcessingTask`
```python
@dataclass
class ProcessingTask:
    id: str
    user_id: int
    task_type: str
    data: Dict[str, Any]
    priority: Priority
    status: TaskStatus
    created_at: datetime
```

#### `MediaFile`
```python
@dataclass
class MediaFile:
    file_id: str
    file_type: str
    file_size: int
    local_path: str
    mime_type: str
    caption: Optional[str]
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest tests/test_telegram_handler.py
pytest tests/test_llm_processor.py
pytest tests/test_integrations.py
```

### Test Configuration
Set up test environment:
```bash
cp .env.example .env.test
# Configure test-specific settings
export TEST_ENV=true
```

## 📊 Monitoring and Logging

### Logging Levels
- **DEBUG**: Detailed debugging information
- **INFO**: General application flow
- **WARNING**: Warning conditions
- **ERROR**: Error conditions
- **CRITICAL**: Critical errors

### Log Files
- `data/logs/app.log` - General application logs
- `data/logs/errors.log` - Error-specific logs
- `data/logs/performance.log` - Performance metrics
- `data/logs/structured.log` - Structured JSON logs

### Health Monitoring
```bash
# Check application health
curl http://localhost:8080/health

# View metrics (if enabled)
curl http://localhost:8090/metrics
```

## 🔒 Security Considerations

### API Key Management
- Store API keys in environment variables
- Use different keys for different environments
- Rotate keys regularly
- Never commit keys to version control

### Data Security
- All sensitive data is encrypted at rest
- Temporary files are automatically cleaned up
- Session data expires automatically
- Rate limiting prevents abuse

### Network Security
- Use HTTPS for all external API calls
- Validate all input data
- Implement proper CORS policies
- Monitor for suspicious activity

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py", "prod"]
```

### Environment-Specific Settings

#### Development
- Debug mode enabled
- Verbose logging
- Mock external services (optional)
- Hot reloading

#### Staging
- Production-like environment
- Full integration testing
- Performance monitoring
- Error reporting

#### Production
- Optimized for performance
- Comprehensive monitoring
- Automated backups
- High availability setup

### Scaling Considerations
- Use Redis for shared caching
- Implement database connection pooling
- Configure load balancing for multiple instances
- Monitor resource usage and scale accordingly

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Set up pre-commit hooks: `pre-commit install`
5. Run tests and ensure they pass
6. Submit a pull request

### Code Standards
- Follow PEP 8 style guide
- Use type hints throughout
- Write comprehensive docstrings
- Maintain test coverage above 90%
- Use meaningful commit messages

### Adding New Features
1. **New Actions**: Extend the `DecisionEngine` class
2. **LLM Providers**: Implement the provider interface
3. **Integrations**: Add new handlers in the `integrations/` directory
4. **Data Sources**: Extend media processing capabilities

## 📝 Additional Integrations

The modular architecture makes it easy to add new integrations:

### Suggested Additional Actions
5. **Slack Integration**: Post processed content to Slack channels
6. **Email Notifications**: Send analysis results via email
7. **Database Updates**: Insert data into various database systems
8. **File Storage**: Upload processed files to cloud storage (AWS S3, Google Drive)
9. **Social Media Posting**: Auto-post to Twitter, LinkedIn, Facebook
10. **CRM Integration**: Add leads and contacts to CRM systems
11. **Analytics Reporting**: Generate and send automated reports
12. **Task Management**: Create tasks in project management tools
13. **Calendar Events**: Schedule events based on processed content
14. **PDF Generation**: Create formatted PDF reports
15. **Translation Services**: Translate content to multiple languages

### Integration Templates
Each new integration should follow this pattern:
```python
class NewServiceHandler:
    async def initialize(self) -> None:
        # Setup service connection
    
    async def process_data(self, data: Dict) -> Result:
        # Process data for the service
    
    async def health_check(self) -> Dict:
        # Check service availability
```

## 📚 Documentation

- [API Reference](docs/api_reference.md)
- [Configuration Guide](docs/configuration.md)
- [Deployment Guide](docs/deployment.md)
- [User Guide](docs/user_guide.md)
- [Development Guide](docs/development.md)

## 🆘 Troubleshooting

### Common Issues

#### Bot Not Responding
1. Check bot token validity
2. Verify network connectivity
3. Check log files for errors
4. Ensure webhook is properly configured

#### LLM Processing Failures
1. Verify API keys are correct
2. Check token limits and quotas
3. Monitor rate limiting
4. Review prompt templates

#### Integration Errors
1. Test API connections manually
2. Verify credentials and permissions
3. Check service status pages
4. Review integration-specific logs

### Support
- Check the [Issues](link-to-issues) page for known problems
- Review [Discussions](link-to-discussions) for community help
- Create a new issue with detailed error information
- Include relevant log excerpts and configuration (without sensitive data)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) community
- OpenAI and Anthropic for LLM APIs
- Contributors and testers
- Open source libraries that make this possible

## 🔄 Changelog

### v0.1.0 (2025-09-16)
- Initial release
- Core Telegram bot functionality
- OpenAI and Anthropic LLM integration
- WooCommerce and Excel integrations
- Comprehensive logging and monitoring
- Multi-level state management
- Advanced error handling and recovery

---

**Built with ❤️ for intelligent automation**
