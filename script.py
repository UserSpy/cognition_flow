"""
Starting from history_modifier and ending in output_modifier, the
functions are declared in the same order that they are called at
generation time.
"""

import gradio as gr
import os
import json
import torch
import sys
from html import unescape
from importlib import import_module
from transformers import LogitsProcessor
from modules.extensions import apply_extensions
from modules import chat, shared
from modules.text_generation import (
    decode,
    encode,
    generate_reply,
)

params = {
    "display_name": "Cognition Flow",
    "is_tab": False,
}

presetPath = ''
nextInstruction = 0
historyModifier = 0
thoughtInput = ''
lastGeneratedResponse = ''
backgroundGenerationHistory = ''
finalGen = 5
genNumber = 0
loopNumber = 0
finalLoop = 5
useContext = True
willGenerate = True
saveInputToHistory = False
saveResponseToHistory = True
previousBotReply = ''
userReply = ''
enabled = True
outputModified = False


class MyLogits(LogitsProcessor):
    """
    Manipulates the probabilities for the next token before it gets sampled.
    Used in the logits_processor_modifier function below.
    """

    def __init__(self):
        pass

    def __call__(self, input_ids, scores):
        # probs = torch.softmax(scores, dim=-1, dtype=torch.float)
        # probs[0] /= probs[0].sum()
        # scores = torch.log(probs / (1 - probs))
        return scores



def toggle_enabled(value):
    global enabled
    enabled = value


def history_modifier(history):
    """
    Modifies the chat history.

    Only used in chat mode.
    """
    return history


def state_modifier(state):
    """
    Modifies the state variable, which is a dictionary containing the input
    values in the UI like sliders and checkboxes.
    """
    if 'new_generation' not in state:
        # Add a new key 'new_generation' and set it to True
        state['new_generation'] = True
        state['stream'] = False
    else:
        state['new_generation'] = False
    return state


def chat_input_modifier(text, visible_text, state):
    """
    Modifies the user input string in chat mode (visible_text).
    You can also modify the internal representation of the user
    input (text) to change how it will appear in the prompt.
    """
    return text, visible_text


def input_modifier(string, state, is_chat=False):
    """
    In default/notebook modes, modifies the whole prompt.

    In chat mode, it is the same as chat_input_modifier but only applied
    to "text", here called "string", and not to "visible_text".
    """
    if (not enabled):
        return string
    if (state['new_generation']):
        global lastGeneratedResponse
        global genNumber
        global nextInstruction
        global historyModifier
        global userReply
        global previousBotReply
        lastGeneratedResponse = ''
        genNumber = 0
        historyModifier = 0
        nextInstruction = 0
        userReply = string
        previousBotReply = unescape(state['history']['visible'][-1][-1])
        thinking_loop(string, state)
        return thoughtInput
        
    return string

def thinking_loop(string, state): #these next functions are custom functions that are not called in order
    """
    Loops text generation onto itself
    """

    global lastGeneratedResponse
    global backgroundGenerationHistory
    global genNumber
    global loopNumber
    global nextInstruction
    global thoughtInput

    
    
    if (saveResponseToHistory):
        backgroundGenerationHistory = backgroundGenerationHistory + ' ' + string
    
    loopNumber = 0
    thoughtInput = string
    while(not (loopNumber == finalLoop)):
        thoughtInput = customInstructions(thoughtInput, backgroundGenerationHistory, previousBotReply, userReply, state)
        if genNumber == finalGen - 1:
                nextInstruction = -1
        loopNumber += 1
        if (saveInputToHistory):
            backgroundGenerationHistory = thoughtInput
        if (willGenerate):
            break
    print('\n' + thoughtInput)

    print("\n-----------------------------------------------------\n")
    print("Generation:", (genNumber + 1))
    
    if ((genNumber == finalGen - 1) or (nextInstruction == -1)):
        genNumber += 1
        return thoughtInput #go back to input_modifier
    
    lastGeneratedResponse = generate(thoughtInput, state)
    thinking_loop(lastGeneratedResponse, state)


def customInstructions(string, backGroundGenerationHistory, previousBotReply, userReply, state):
    global historyModifier
    global nextInstruction
    global finalGen
    global finalLoop
    global willGenerate
    global saveInputToHistory
    global saveResponseToHistory
    global useContext
    global outputModified
    # Load JSON file
    
    try:
        with open(presetPath, 'r') as f:
            config = json.load(f)
    except:
        print("Could not load JSON preset")
        return string
    try:
        if (genNumber == 0):
            finalGen = config['max_gen_num']
            finalLoop = config['max_loop_num']
    except:
        print("Could not find all max values in JSON, defaulting to 5")
    # Find the function configuration matching nextInstruction
    function_config = None
    for func in config['flow_functions']:
        if func['identifier'] == nextInstruction:
            function_config = func
            break
    
    outputModified = False
    if (function_config is None):
        print ("Instruction function with identifier", nextInstruction, "not found. Executing final output.")
        nextInstruction = -1
        return string
    elif (nextInstruction == -1):
        outputModified = True
        
    
    
    
    # Dynamically import and run the script
    try:
        script_path = function_config['script_path']
        module_name = script_path.replace('.py', '').replace('/', '.')
        module = import_module(module_name)
        # Assuming the function is called main
        retString, retValue = module.main(string, backGroundGenerationHistory, previousBotReply, userReply, state)
    except:
        print('Error importing python script from path: ', script_path)
        return string

    if module_name in sys.modules:
        del sys.modules[module_name]

    # Update nextInstruction and historyModifier based on return code
    returnCode = None
    for code in function_config['return_codes']:
        if code['code'] == retValue:
            returnCode = code
            nextInstruction = code['mapped_to_instruction']
            willGenerate = code['willGenerate']
            saveInputToHistory = code['saveInputToHistory']
            saveResponseToHistory = code['saveResponseToHistory']
            useContext = code['useContext']
            break
    if (returnCode is None) and not (nextInstruction == -1):
        print('\nReturn Code', retValue, 'Not Defined: May cause unexpected behavior\n')
    print('\nReturn Code:', returnCode, "\n")

    return retString
    
def generate(input, state):
    global genNumber
    retString = None
    if (useContext):
        prompt = apply_extensions('custom_generate_chat_prompt', input, state)
    else:
        prompt = input
        print("not using context, input is", prompt)
    errorCapture = None
    genNumber += 1
    state['stream'] = True
    for i, history in enumerate(chat.generate_reply(prompt, state)):
        if errorCapture is not None:
            retString = errorCapture     # fixes weird bug where history is set to the previous loops final generation
        errorCapture = history
    print("\n", retString, "\n")
    print("\n-----------------------------------------------------\n")
    return retString #go back to thinking loop    

#end of custom functions, go back to normal flow
    
def bot_prefix_modifier(string, state):
    """
    Modifies the prefix for the next bot reply in chat mode.
    By default, the prefix will be something like "Bot Name:".
    """
    return string


def tokenizer_modifier(state, prompt, input_ids, input_embeds):
    """
    Modifies the input ids and embeds.
    Used by the multimodal extension to put image embeddings in the prompt.
    Only used by loaders that use the transformers library for sampling.
    """
    return prompt, input_ids, input_embeds


def logits_processor_modifier(processor_list, input_ids):
    """
    Adds logits processors to the list, allowing you to access and modify
    the next token probabilities.
    Only used by loaders that use the transformers library for sampling.
    """
    processor_list.append(MyLogits())
    return processor_list


def output_modifier(string, state, is_chat=False):
    """
    Modifies the LLM output before it gets presented.

    In chat mode, the modified version goes into history['visible'],
    and the original version goes into history['internal'].
    """
    if (not enabled):
        return string
    global previousBotReply
    
    if ((genNumber == 0) or (nextInstruction == -1)):
        print("\n-----------------------------------------------------\n")
        

        if (nextInstruction == -1):
            temp = customInstructions(string, backgroundGenerationHistory, previousBotReply, userReply, state)
            if (outputModified):
                string = temp
            
        return string
    


    return string #lastGeneratedResponse??



def custom_generate_chat_prompt(user_input, state, **kwargs):
    """
    Replaces the function that generates the prompt from the chat history.
    Only used in chat mode.
    """
    result = chat.generate_chat_prompt(user_input, state, **kwargs)
    return result


def custom_css():
    """
    Returns a CSS string that gets appended to the CSS for the webui.
    """
    return ''


def custom_js():
    """
    Returns a javascript string that gets appended to the javascript
    for the webui.
    """
    return ''


def setup():
    """
    Gets executed only once, when the extension is imported.
    """
    pass


def ui():
    """
    Gets executed when the UI is drawn. Custom gradio elements and
    their corresponding event handlers should be defined here.

    To learn about gradio components, check out the docs:
    https://gradio.app/docs/
    """
    global dropdown_menu, ui_container, preset_name_textbox

    with gr.Accordion("Cognition Flow", open=True, elem_id='cognition_flow'):
        enabled_checkbox = gr.Checkbox(value=enabled, label='Enabled')
        enabled_checkbox.change(
            lambda x: toggle_enabled(x), enabled_checkbox, None)
        preset_choices = list_flow_presets()
        default_choice = preset_choices[0] if preset_choices else None
        dropdown_menu = gr.Dropdown(
            choices=preset_choices, label='Flow Preset', value=default_choice)
        dropdown_menu.change(lambda x: load_preset(x), dropdown_menu, None)

        preset_name_textbox = gr.Textbox(label="Preset Name", visible=False)
        ui_container = gr.Column()
    load_preset(default_choice)


def load_preset(selected_preset):
    global presetPath
    global nextInstruction
    global historyModifier
    historyModifier = 0
    nextInstruction = 0
    presetPath = f"extensions/cognition_flow/flow_presets/{selected_preset}"
    

    with open(presetPath, 'r') as f:
        data = json.load(f)
        print("Loaded preset name: ", data.get('preset_name', ''))
    # update_ui_with_json(data)





def list_flow_presets():
    preset_path = "extensions/cognition_flow/flow_presets"
    return [f for f in os.listdir(preset_path) if f.endswith('.json')]


def update_preset():
    selected_preset = dropdown_menu.get_value()
    preset_path = f"extensions/cognition_flow/flow_presets/{selected_preset}"
    with open(preset_path, 'r+') as f:
        data = json.load(f)
        f.seek(0)
        json.dump(data, f)
        f.truncate()







