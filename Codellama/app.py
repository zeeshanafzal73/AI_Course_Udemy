import requests
import json
import gradio as gr

url = "http://localhost:11434/api/generate"
headers = {
    'Content-Type': 'application/json'
}

# Initialize history as an empty list for each interaction session
def generate_response(prompt, history=[]):
    # Append the prompt to history
    history.append(prompt)
    final_prompt = "\n".join(history)

    data = {
        "model": "codeguru",
        "prompt": final_prompt,
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))

        if response.status_code == 200:
            # Parse the response data
            response_data = json.loads(response.text)
            actual_response = response_data.get('response', 'No response found')
            return actual_response
        else:
            return f"Error: {response.status_code} - {response.text}"

    except requests.exceptions.RequestException as e:
        # Handle request exceptions like timeouts, connection errors, etc.
        return f"Request failed: {str(e)}"


interface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=4, placeholder="Enter your Prompt"),
    outputs="text"
)

interface.launch()
