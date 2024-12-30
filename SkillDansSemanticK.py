import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, AzureChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

kernel = Kernel()

# Prepare OpenAI service using credentials stored in the `.env` file
#kernel.add_service(OpenAIChatCompletion(service_id="chat-gpt", ai_model_id="gpt-3.5-turbo"))
service_id = "azure-gpt-35-turbo-16k"
# Prepare Azure service using credentials stored in the `.env` file
kernel.add_service(
    AzureChatCompletion(
        deployment_name="gpt-35-turbo-16k",
        service_id=service_id,
        endpoint=os.getenv("AZURE_ENDPOINT"),
        api_key=os.getenv("AZURE_API_KEY"),
    )
)


# Define the request settings
req_settings = kernel.get_prompt_execution_settings_from_service_id(service_id)
req_settings.max_tokens = 1000
req_settings.temperature = 0.7
req_settings.top_p = 0.8

prompt = """
1) A robot may not injure a human being or, through inaction,
allow a human being to come to harm.

2) A robot must obey orders given it by human beings except where
such orders would conflict with the First Law.

3) A robot must protect its own existence as long as such protection
does not conflict with the First or Second Law.

Give me the TLDR in exactly 5 words."""

prompt_template_config = PromptTemplateConfig(
    template=prompt,
    name="tldr",
    template_format="semantic-kernel",
    execution_settings=req_settings,
)

function = kernel.add_function(
    function_name="tldr_function",
    plugin_name="tldr_plugin",
    prompt_template_config=prompt_template_config,
    execution_settings=req_settings,
)

# Run your prompt
# Note: functions are run asynchronously
async def main():
    result = await kernel.invoke(function)
    print(result) # => Robots must not harm humans.




# Create a reusable function summarize function
summarize = kernel.add_function(
    function_name="tldr_function",
    plugin_name="tldr_plugin",
    prompt="{{$input}}\n\nOne line TLDR with the fewest words.",
    execution_settings=req_settings,
)

async def summarize_laws():
    # Summarize the laws of thermodynamics
    print(await kernel.invoke(summarize, input="""
    1st Law of Thermodynamics - Energy cannot be created or destroyed.
    2nd Law of Thermodynamics - For a spontaneous process, the entropy of the universe increases.
    3rd Law of Thermodynamics - A perfect crystal at zero Kelvin has zero entropy."""))


if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(summarize_laws())
# If running from a jupyter-notebook:
# await main()
# await summarize_laws()

# Output:
# > Energy conserved, entropy increases, zero entropy at 0K.
# > Objects move in response to forces.
# > Gravitational force between two point masses is inversely proportional to the square of the distance between them.