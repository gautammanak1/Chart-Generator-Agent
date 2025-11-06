import os
import re
import time
import requests
from datetime import datetime, timezone
from typing import Tuple
from uuid import uuid4
import io
import base64

from uagents import Agent, Context, Protocol
from uagents.setup import fund_agent_if_low
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    StartSessionContent,
    TextContent,
    chat_protocol_spec,
)
from dotenv import load_dotenv

load_dotenv()

# Initialize the agent
agent = Agent(
    name="chart_generator_agent",
    mailbox=True,
    port=8000,
    seed="chart-generator-seed-phrase-change-in-production",
)

# Fund the agent if needed
fund_agent_if_low(agent.wallet.address())

# Initialize the chat protocol with the standard chat spec
chat_proto = Protocol(spec=chat_protocol_spec)


def get_asi_chart_code(api_key: str, prompt: str) -> str:
    """Generate Python chart code using ASI 1 API."""
    url = "https://api.asi1.ai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    system_prompt = """You are a Python chart generation expert. Generate Python code to create various types of charts based on user requests.

Supported Chart Types:
- Area charts: Use plt.fill_between() or plt.stackplot()
- Bar charts: Use plt.bar() or plt.barh()
- Box plots: Use plt.boxplot()
- Column charts: Use plt.bar() (vertical bars)
- Dual axes charts: Use plt.twinx() or plt.twiny()
- Histograms: Use plt.hist()
- Line charts: Use plt.plot()
- Pie charts: Use plt.pie()
- Radar charts: Use matplotlib with polar coordinates (plt.subplot(projection='polar'))
- Sankey diagrams: Use matplotlib.sankey.Sankey correctly
- Scatter plots: Use plt.scatter()
- Treemap: Use squarify library or matplotlib patches
- Venn diagrams: Use matplotlib_venn or matplotlib patches
- Violin plots: Use plt.violinplot()
- Word clouds: Use wordcloud.WordCloud() - import wordcloud; wc = wordcloud.WordCloud(); wc.generate_from_text() or wc.generate_from_frequencies()
- Network graphs: Use networkx with matplotlib - import networkx as nx; G = nx.Graph(); nx.draw(G, ax=plt.gca())
- Flowcharts: Use matplotlib patches and annotations
- Funnel charts: Use plt.barh() with decreasing widths
- Mind maps: Use networkx with hierarchical layout
- Organization charts: Use networkx with hierarchical layout
- Fishbone diagrams: Use matplotlib with arrows and text annotations
- Liquid charts: Use matplotlib with circle patches and fill
- 3D surface charts: Use mpl_toolkits.mplot3d.Axes3D

Requirements:
- Set matplotlib backend to 'Agg' (non-interactive): import matplotlib; matplotlib.use('Agg')
- Import required libraries: matplotlib.pyplot as plt, numpy as np, wordcloud (for word clouds), networkx as nx (for graphs)
- Do NOT use plt.savefig() - just create the charts
- Do NOT use plt.show() - charts will be captured automatically
- Include proper titles, axis labels, and formatting
- Output ONLY the Python code in a code block
- No explanations, no comments, just raw Python code
- Generate realistic sample data if specific data is not provided
- Do NOT save files - just create matplotlib figures
- Always initialize variables properly before using them
- For word clouds: Use wordcloud.WordCloud().generate_from_frequencies() or .generate()
- For network graphs: Use networkx.Graph() or networkx.DiGraph() with nx.draw()
- For Sankey: Sankey() returns a Sankey object, not a list"""
    
    enhanced_prompt = f"""Generate Python code using matplotlib to create: {prompt}

Requirements:
- Use matplotlib with Agg backend: import matplotlib; matplotlib.use('Agg')
- Do NOT use plt.savefig() - just create the chart (plt.plot(), plt.bar(), etc.)
- Do NOT save any files
- Create realistic sample data if specific data is not provided
- Include proper chart titles, axis labels, and formatting
- Output only the Python code in a code block"""
    
    data = {
        "model": "asi1-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": enhanced_prompt}
        ]
    }
    
    try:
        response = requests.post(url, json=data, headers=headers, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"API Error: {str(e)}"


def extract_code(response_text: str) -> str:
    """Extract Python code from LLM response."""
    # Try to extract code from markdown code blocks
    code_match = re.search(r"```python\n(.*?)```", response_text, re.DOTALL)
    if code_match:
        code = code_match.group(1)
    else:
        # Try without language specification
        code_match = re.search(r"```\n(.*?)```", response_text, re.DOTALL)
        if code_match:
            code = code_match.group(1)
        else:
            # Try to find code block with any language
            code_match = re.search(r"```[\w]*\n(.*?)```", response_text, re.DOTALL)
            if code_match:
                code = code_match.group(1)
            else:
                # Use the entire response if no code block found
                code = response_text
    
    # Clean up the code
    code = code.strip()
    
    # Remove any leading/trailing markdown artifacts
    code = re.sub(r'^```python\s*', '', code)
    code = re.sub(r'^```\s*', '', code)
    code = re.sub(r'\s*```$', '', code)
    
    return code.strip()


def upload_to_imgbb(image_data: bytes) -> str:
    """Upload image to imgbb (free image hosting) and return URL."""
    try:
        # Convert image to base64
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        # imgbb API endpoint
        url = "https://api.imgbb.com/1/upload"
        
        api_key = os.getenv("IMGBB_API_KEY", "5fc10a16325262d93d1e884c7c0e6eae")
        
        if api_key:
            data = {
                "key": api_key,
                "image": image_b64
            }
            response = requests.post(url, data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    return result.get("data", {}).get("url", "")
        
        # Fallback: Try imgur (anonymous upload)
        return upload_to_imgur(image_data)
    except Exception as e:
        # Fallback to imgur
        return upload_to_imgur(image_data)


def upload_to_imgur(image_data: bytes) -> str:
    """Upload to imgur (anonymous, free image hosting)."""
    try:
        url = "https://api.imgur.com/3/image"
        headers = {
            "Authorization": "Client-ID 546c25a59c58ad7"  
        }
        data = {
            "image": base64.b64encode(image_data).decode('utf-8')
        }
        response = requests.post(url, headers=headers, data=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                return result.get("data", {}).get("link", "")
    except Exception as e:
        pass
    
    # If all fails, return empty (will show error message)
    return ""


def execute_and_upload_chart(code: str, ctx: Context = None) -> Tuple[bool, str, list[str]]:
    """Execute Python code locally, capture charts, and upload to temporary image hosting."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    
    chart_urls = []
    
    try:
        # Create a namespace for code execution with common libraries
        namespace = {
            'plt': plt,
            'np': np,
            'numpy': np,
            'matplotlib': matplotlib,
            '__builtins__': __builtins__
        }
        
        # Try to import additional libraries if available
        try:
            import wordcloud
            namespace['wordcloud'] = wordcloud
            namespace['WordCloud'] = wordcloud.WordCloud
        except ImportError:
            pass
        
        try:
            import networkx as nx
            namespace['networkx'] = nx
            namespace['nx'] = nx
        except ImportError:
            pass
        
        try:
            import pandas as pd
            namespace['pd'] = pd
            namespace['pandas'] = pd
        except ImportError:
            pass
        
        try:
            import scipy
            namespace['scipy'] = scipy
        except ImportError:
            pass
        
        try:
            from mpl_toolkits.mplot3d import Axes3D
            namespace['Axes3D'] = Axes3D
        except ImportError:
            pass
        
        try:
            from matplotlib_venn import venn2, venn3, venn2_circles, venn3_circles
            namespace['venn2'] = venn2
            namespace['venn3'] = venn3
            namespace['venn2_circles'] = venn2_circles
            namespace['venn3_circles'] = venn3_circles
            namespace['matplotlib_venn'] = __import__('matplotlib_venn')
        except ImportError:
            pass
        
        # Execute the code
        exec(code, namespace)
        
        # Check for WordCloud objects in namespace (they need special handling)
        wordcloud_objects = []
        for key, value in namespace.items():
            if hasattr(value, '__class__') and 'WordCloud' in str(type(value)):
                try:
                    # Convert WordCloud to image
                    img_array = value.to_array()
                    if img_array is not None:
                        wordcloud_objects.append(img_array)
                except:
                    pass
        
        # Check if there are any open figures
        fig_nums = plt.get_fignums()
        
        # If we have wordcloud objects but no figures, create a figure from wordcloud
        if wordcloud_objects and not fig_nums:
            for wc_img in wordcloud_objects:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(wc_img, interpolation='bilinear')
                ax.axis('off')
                plt.tight_layout(pad=0)
            # Update fig_nums after creating wordcloud figures
            fig_nums = plt.get_fignums()
        
        if fig_nums:
            # Capture figures directly to memory (BytesIO) - NO LOCAL FILES
            for i, fig_num in enumerate(fig_nums):
                try:
                    plt.figure(fig_num)
                    
                    # Save figure directly to BytesIO buffer (in memory)
                    img_buffer = io.BytesIO()
                    plt.savefig(img_buffer, format='png', bbox_inches='tight')
                    img_buffer.seek(0)  # Reset buffer position
                    img_data = img_buffer.read()
                    
                    # Upload to temporary image hosting
                    image_url = upload_to_imgbb(img_data)
                    
                    if image_url:
                        chart_urls.append(image_url)
                        if ctx:
                            ctx.logger.info(f"Uploaded chart to temporary hosting: {image_url}")
                    else:
                        if ctx:
                            ctx.logger.warning(f"Failed to upload chart {i} to image hosting")
                    
                    # Close the buffer
                    img_buffer.close()
                    
                except Exception as upload_error:
                    if ctx:
                        ctx.logger.error(f"Error uploading figure {fig_num}: {str(upload_error)}")
        
        # If no figures found, check if code saved files directly (we'll read and delete them)
        if not fig_nums:
            import glob
            saved_files = glob.glob('chart_*.png') + [f for f in glob.glob('*.png') if f.endswith('.png')]
            
            for saved_file in saved_files:
                # Skip if file is too old (might be from previous runs)
                try:
                    file_time = os.path.getmtime(saved_file)
                    if time.time() - file_time > 60:  # Skip files older than 60 seconds
                        continue
                except:
                    pass
                
                try:
                    # Read the image file
                    with open(saved_file, 'rb') as f:
                        img_data = f.read()
                    
                    # Upload to temporary image hosting
                    image_url = upload_to_imgbb(img_data)
                    
                    if image_url:
                        chart_urls.append(image_url)
                        if ctx:
                            ctx.logger.info(f"Uploaded chart to temporary hosting: {image_url}")
                    
                    # Delete the file immediately after upload
                    try:
                        os.remove(saved_file)
                    except:
                        pass
                        
                except Exception as upload_error:
                    if ctx:
                        ctx.logger.error(f"Error uploading {saved_file}: {str(upload_error)}")
                    # Try to delete even if upload failed
                    try:
                        os.remove(saved_file)
                    except:
                        pass
        
        # Close all matplotlib figures
        plt.close('all')
        
        if chart_urls:
            success_msg = f"✅ Successfully generated {len(chart_urls)} chart(s)!\n\n"
            return True, success_msg, chart_urls
        else:
            return True, "Code executed successfully but no charts were generated. Make sure to use plt.show() or plt.savefig() to create charts.", []
            
    except AttributeError as attr_err:
        # Clean up on error - close any open figures
        plt.close('all')
        
        error_msg = f"Attribute error in code: {str(attr_err)}\n\n"
        error_msg += "This usually means the code is trying to access an attribute that doesn't exist.\n"
        error_msg += "Common causes:\n"
        error_msg += "- Incorrect API usage (e.g., accessing .texts on a list instead of a Sankey object)\n"
        error_msg += "- Wrong variable type or missing initialization\n"
        
        if ctx:
            ctx.logger.error(f"AttributeError: {str(attr_err)}")
        return False, error_msg, []
            
    except Exception as e:
        # Clean up on error - close any open figures
        plt.close('all')
        
        error_msg = f"Error executing code: {str(e)}\n\n"
        error_msg += "Common issues:\n"
        error_msg += "- Check if all required libraries are imported\n"
        error_msg += "- Verify variable names and data types\n"
        error_msg += "- Ensure matplotlib figures are created before saving\n"
        error_msg += "- For Sankey diagrams, use matplotlib.sankey.Sankey correctly\n"
        
        if ctx:
            ctx.logger.error(f"Execution error: {str(e)}")
        return False, error_msg, []


# Utility function to wrap plain text into a ChatMessage
def create_text_chat(text: str, end_session: bool = False) -> ChatMessage:
    content = [TextContent(type="text", text=text)]
    return ChatMessage(
        timestamp=datetime.now(timezone.utc),
        msg_id=uuid4(),
        content=content,
    )


# Handle incoming chat messages
@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    ctx.logger.info(f"Received message from {sender}")

    # Always send back an acknowledgement when a message is received
    await ctx.send(sender, ChatAcknowledgement(
        timestamp=datetime.now(timezone.utc),
        acknowledged_msg_id=msg.msg_id
    ))

    # Process each content item inside the chat message
    for item in msg.content:
        # Marks the start of a chat session
        if isinstance(item, StartSessionContent):
            ctx.logger.info(f"Session started with {sender}")
            welcome_msg = create_text_chat(
                "👋 Hello! I'm your Chart Generator Agent powered by ASI 1.\n"
                "Send me a description of the chart you want, and I'll create it for you!\n\n"
                "Examples:\n"
                "- 'Create a bar chart showing sales by month'\n"
                "- 'Make a line chart of temperature over time'\n"
                "- 'Generate a pie chart of market share'"
            )
            await ctx.send(sender, welcome_msg)

        # Handles plain text messages (from another agent or ASI:One)
        elif isinstance(item, TextContent):
            user_prompt = item.text.strip()
            ctx.logger.info(f"Text message from {sender}: {user_prompt}")

            # Send initial processing message
            processing_msg = create_text_chat("🔄 Processing your request... Generating chart code with ASI 1...")
            await ctx.send(sender, processing_msg)

            # Get API key
            api_key = os.getenv("ASI_ONE_API_KEY")
            if not api_key:
                error_msg = create_text_chat(
                    "❌ Error: ASI_ONE_API_KEY environment variable not set. "
                    "Please set it in your .env file."
                )
                await ctx.send(sender, error_msg)
                continue

            try:
                # Generate chart code using ASI 1
                code_response = get_asi_chart_code(api_key, user_prompt)
                
                if code_response.startswith("Error"):
                    error_msg = create_text_chat(f"❌ Error generating code: {code_response}")
                    await ctx.send(sender, error_msg)
                    continue

                # Extract code from response
                code = extract_code(code_response)
                ctx.logger.info(f"Extracted code length: {len(code)} characters")
                
                # Validate code syntax before execution
                try:
                    compile(code, '<string>', 'exec')
                except SyntaxError as syntax_err:
                    error_msg = create_text_chat(
                        f"❌ Syntax error in generated code:\n```\n{str(syntax_err)}\n```\n\n"
                        f"Generated code preview:\n```python\n{code[:500]}\n```"
                    )
                    await ctx.send(sender, error_msg)
                    continue

                # Execute code and upload to temporary hosting
                success, result_msg, chart_urls = execute_and_upload_chart(code, ctx)

                # Send result message with markdown image links
                if success:
                    if chart_urls:
                        result_text = result_msg
                        # Add markdown image links for each chart
                        result_text += "**Charts:**\n\n"
                        for i, url in enumerate(chart_urls, 1):
                            result_text += f"![Chart {i}]({url})\n\n"
                    else:
                        result_text = f"ℹ️ {result_msg}"
                else:
                    result_text = f"❌ {result_msg}"
                
                response_message = create_text_chat(result_text)
                await ctx.send(sender, response_message)
                
            except Exception as e:
                ctx.logger.error(f"Error in chart generation: {str(e)}")
                error_msg = create_text_chat(f"❌ Unexpected error: {str(e)}")
                await ctx.send(sender, error_msg)

        # Marks the end of a chat session
        elif isinstance(item, EndSessionContent):
            ctx.logger.info(f"Session ended with {sender}")
            goodbye_msg = create_text_chat("👋 Session ended. Goodbye!")
            await ctx.send(sender, goodbye_msg)

        # Catches anything unexpected
        else:
            ctx.logger.info(f"Received unexpected content type from {sender}")


# Handle acknowledgements for messages this agent has sent out
@chat_proto.on_message(ChatAcknowledgement)
async def handle_acknowledgement(ctx: Context, sender: str, msg: ChatAcknowledgement):
    ctx.logger.info(f"Received acknowledgement from {sender} for message {msg.acknowledged_msg_id}")


# Include the chat protocol and publish the manifest to Agentverse
agent.include(chat_proto, publish_manifest=True)


if __name__ == "__main__":
    agent.run()
