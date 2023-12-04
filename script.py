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
    if 'new_generation' not in state: #used to control flow in other parts of the script
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
        global lastGeneratedResponse #setting initial values for global variables
        global genNumber
        global nextInstruction
        global historyModifier
        global userReply
        global previousBotReply
        global backgroundGenerationHistory
        global thoughtInput
        thoughtInput = ''
        backgroundGenerationHistory = ''
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
    Loops text generation onto itself and modifies global variables based on loaded functions
    """

    global lastGeneratedResponse
    global backgroundGenerationHistory
    global genNumber
    global loopNumber
    global nextInstruction
    global thoughtInput

    
    
    if (saveResponseToHistory and not state['new_generation']):
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


def customInstructions(string, backGroundGenerationHistory, previousBotReply, userReply, state): #loads a JSON configuration and executes a script based on the current state and input
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

    return retString
    
def generate(input, state):
    #Handles the text generation process. It uses the provided state and input to generate a response, taking into account whether to use context or not.
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
    global outputModified
    global nextInstruction
    if ((genNumber == 0) or (nextInstruction == -1)):
        print("\n-----------------------------------------------------\n")
        

        if (nextInstruction == -1):
            temp = customInstructions(string, backgroundGenerationHistory, previousBotReply, userReply, state)
            nextInstruction = 0
            if (outputModified):
                string = temp
                outputModified = False
            
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
    global preset_dropdown, ui_container, preset_name_textbox

    with gr.Accordion("Cognition Flow", open=True, elem_id='cognition_flow'):
        enabled_checkbox = gr.Checkbox(value=enabled, label='Enabled')
        enabled_checkbox.change(
            lambda x: toggle_enabled(x), enabled_checkbox, None)
        preset_choices = list_flow_presets()
        default_choice = preset_choices[0] if preset_choices else None
        with gr.Column():
            preset_dropdown = gr.Dropdown(choices=preset_choices, label='Flow Preset', allow_custom_value=True)
            refresh_presets = gr.Button("Refresh")
        refresh_presets.click(refresh_ui, outputs=[preset_dropdown])
        preset_dropdown.update(default_choice)
        preset_name_textbox = gr.Textbox(label="Preset Name", visible=False)
        ui_container = gr.Column()
        new_preset_name_textbox = gr.Textbox(label="New Preset Filename")
        create_preset_button = gr.Button("Create New Preset")
        create_preset_button.click(create_new_preset, inputs=[new_preset_name_textbox], outputs=[preset_dropdown, preset_dropdown, new_preset_name_textbox])
        with gr.Accordion("Preset Editor", open=False, elem_id='preset_editor'):
            maxGen = gr.Number(label="Maximum Generations", interactive=True)
            maxLoop = gr.Number(label="Maximum Non-Generation Loops", interactive=True)
            with gr.Row():
                with gr.Column():
                    preset_function_dropdown = gr.Dropdown(
                        choices=['Please Select a Preset'], label='Preset Function', interactive=True, allow_custom_value=True)  
                    with gr.Row():
                        save_function_button = gr.Button("Save Function")
                        new_function_button = gr.Button("Create New Function")
                    functionName = gr.Textbox(label="Function Name", interactive=True)
                    funcIdentifier = gr.Number(label="Identifier (0 is called first, -1 optionally called last)", interactive=True)
                
                    flowfunctionDropdown = gr.Dropdown(choices=list_flow_functions(), label='Script', interactive=True, allow_custom_value=True)

                    refresh_functions = gr.Button("Refresh")
                refresh_functions.click(refreshFunctions, outputs=[flowfunctionDropdown])
                with gr.Column():
                    function_return_dropdown = gr.Dropdown(choices=['Please Select Preset Function'], label='Return Code', interactive=True, allow_custom_value=True)
                    with gr.Row():
                        save_return_button = gr.Button("Save Return Code")
                        new_return_button = gr.Button("Create New Return Code")
                    codeInput = gr.Textbox(label="Code", interactive=True)
                    instructionMappedTo = gr.Number(label="Mapped to Function Identifier (-1 for final behavior):", interactive=True)
                    generateCheckbox = gr.Checkbox(label='Generate Text using Function Output', interactive=True)
                    saveInputCheckbox = gr.Checkbox(label='Save Function Output as the Temporary History', interactive=True)
                    saveResponseCheckbox = gr.Checkbox(label='Append Text Generated after Function to the Temporary History', interactive=True)
                    useContextCheckbox = gr.Checkbox(label='Use the webui\'s history and character context', interactive=True)
            
    
        maxGen.change(lambda x: change_maxGen(x), maxGen)
        maxLoop.change(lambda x: change_maxLoop(x), maxLoop)
        preset_dropdown.change(lambda x: load_preset(x), preset_dropdown, outputs=[preset_function_dropdown, preset_function_dropdown, maxGen, maxLoop])
        preset_function_dropdown.change(lambda x: update_function_information(x), preset_function_dropdown, outputs=[functionName, funcIdentifier, flowfunctionDropdown, function_return_dropdown, function_return_dropdown])
        function_return_dropdown.change(lambda x, y: update_return_information(x, y), inputs=[preset_function_dropdown, function_return_dropdown], outputs=[codeInput, instructionMappedTo, generateCheckbox, saveInputCheckbox, saveResponseCheckbox, useContextCheckbox])
        save_function_button.click(lambda x, y, z, a: functionChange(x, y, z, a), inputs = [preset_function_dropdown, functionName, funcIdentifier, flowfunctionDropdown], outputs=[preset_function_dropdown, preset_function_dropdown])
        new_function_button.click(lambda y, z, a: functionNew(y, z, a), inputs =[functionName, funcIdentifier, flowfunctionDropdown], outputs=[preset_function_dropdown, preset_function_dropdown])
        save_return_button.click(lambda a, b, c, d, e, f, g, h: returnChange(a, b, c, d, e, f, g, h), inputs=[preset_function_dropdown, function_return_dropdown, codeInput, instructionMappedTo, generateCheckbox, saveInputCheckbox, saveResponseCheckbox, useContextCheckbox], outputs=[function_return_dropdown, function_return_dropdown])
        new_return_button.click(
    lambda a, b, c, d, e, f, g: returnNew(a, b, c, d, e, f, g), 
    inputs=[preset_function_dropdown, codeInput, instructionMappedTo, generateCheckbox, saveInputCheckbox, saveResponseCheckbox, useContextCheckbox],
    outputs=[function_return_dropdown, function_return_dropdown]
)
        
    load_preset(default_choice)

def change_maxGen(new_max_gen):
    # Load the JSON data
    with open(presetPath, 'r') as file:
        data = json.load(file)

    # Update the max_gen_num value
    data['max_gen_num'] = int(new_max_gen)

    # Save the updated data back to JSON file
    with open(presetPath, 'w') as file:
        json.dump(data, file, indent=4)

def change_maxLoop(new_max_loop):
    # Load the JSON data
    with open(presetPath, 'r') as file:
        data = json.load(file)

    # Update the max_loop_num value
    data['max_loop_num'] = int(new_max_loop)

    # Save the updated data back to JSON file
    with open(presetPath, 'w') as file:
        json.dump(data, file, indent=4)

def returnChange(currentFunction, oldCode, newCode, mapped_to_instruction, willGenerate, saveInputToHistory, saveResponseToHistory, useContext):
    if oldCode is None:
        return None, None
    with open(presetPath, 'r') as file:
        data = json.load(file)
    identifier = currentFunction.split(': ')[0]
    for func in data['flow_functions']:
        if str(func['identifier']) == identifier:
        
            for code in func['return_codes']:
                if code['code'] == oldCode:
                    code['code'] = int(newCode)
                    code["mapped_to_instruction"] = mapped_to_instruction
                    code["willGenerate"] = willGenerate
                    code["saveInputToHistory"] = saveInputToHistory
                    code["saveResponseToHistory"] = saveResponseToHistory
                    code ["useContext"] = useContext
                    
            return_codes = [code['code'] for code in func['return_codes']]
    
    with open(presetPath, 'w') as file:
        json.dump(data, file, indent=4)
        
    return_codes.insert(0, "Please Select This First")
    return gr.Dropdown.update(choices=return_codes), return_codes[0]

def returnNew(currentFunction, newCode, mapped_to_instruction, willGenerate, saveInputToHistory, saveResponseToHistory, useContext):
    # Load JSON data
    with open(presetPath, 'r') as file:
        data = json.load(file)

    identifier = currentFunction.split(': ')[0]
    
     # Convert None values from checkboxes to False
    willGenerate = False if willGenerate is None else willGenerate
    saveInputToHistory = False if saveInputToHistory is None else saveInputToHistory
    saveResponseToHistory = False if saveResponseToHistory is None else saveResponseToHistory
    useContext = False if useContext is None else useContext

    # Find the function and append the new return code
    for func in data['flow_functions']:
        if str(func['identifier']) == identifier.strip():
            new_return_code = {
                "code": int(newCode),
                "mapped_to_instruction": mapped_to_instruction,
                "willGenerate": willGenerate,
                "saveInputToHistory": saveInputToHistory,
                "saveResponseToHistory": saveResponseToHistory,
                "useContext": useContext
            }
            func['return_codes'].append(new_return_code)
            break

    # Save the updated data back to JSON file
    with open(presetPath, 'w') as file:
        json.dump(data, file, indent=4)

    return_codes = [code['code'] for func in data['flow_functions'] if str(func['identifier']) == identifier.strip() for code in func['return_codes']]
    return_codes.insert(0, "Please Select This First")
    return gr.Dropdown.update(choices=return_codes), return_codes[0]

def functionChange(currentFunction, name, newIdentifier, script):
    if currentFunction is not None:
        oldIdentifier, oldName = currentFunction.split(': ')
    else:
        return None, None
    
    with open(presetPath, 'r') as file:
        data = json.load(file)

    for function in data["flow_functions"]:
            if str(function["identifier"]) == oldIdentifier.strip():
                function["name"] = name
                function["identifier"] = newIdentifier
                function["script_path"] = "extensions/cognition_flow/flow_functions/" + script
                break
    
    with open(presetPath, 'w') as file:
        json.dump(data, file, indent=4)
    choices = list_preset_functions()
    choices.insert(0, "Please Select This First")
    return gr.Dropdown.update(choices=choices), str(newIdentifier) + ': ' + name


def functionNew(name, newIdentifier, script):
   
    with open(presetPath, 'r') as file:
        data = json.load(file)

    if name is None or newIdentifier is None:
        return None, None
    
    # Create new function entry
    new_function = {
        "name": name,
        "identifier": newIdentifier,
        "script_path": "extensions/cognition_flow/flow_functions/" + script,
        "return_codes": []  # Assuming empty return codes initially
    }

    # Append the new function to the list
    data["flow_functions"].append(new_function)

    # Save the updated data back to JSON file
    with open(presetPath, 'w') as file:
        json.dump(data, file, indent=4)
    
    choices = list_preset_functions()
    choices.insert(0, "Please Select This First")
    return gr.Dropdown.update(choices=choices), str(newIdentifier) + ': ' + name

def load_preset(selected_preset):
    if selected_preset is not None:
        global presetPath
        global nextInstruction
        global historyModifier
        historyModifier = 0
        nextInstruction = 0
        presetPath = f"extensions/cognition_flow/flow_presets/{selected_preset}"
        maxGen = 5
        maxLoop = 5
        
        with open(presetPath, 'r') as f:
            data = json.load(f)
            print("Loaded preset name: ", data.get('preset_name', ''))
            maxGen = data.get('max_gen_num', 5)
            maxLoop = data.get('max_loop_num', 5)
        # update_ui_with_json(data)
        updated_preset_functions = list_preset_functions()
        updated_preset_functions.insert(0, "Please Select This First")
        return gr.Dropdown.update(choices=updated_preset_functions), gr.Dropdown.update(value=updated_preset_functions[0]), maxGen, maxLoop
    else:
        print("No preset selected")


def update_function_information(selected_function):
    with open(presetPath, 'r') as f:
        data = json.load(f)
        if selected_function is None:
            # Handle the None case, maybe return some default values or error message
            return None, None, None, None, None
        identifier = selected_function.split(': ')[0]

    # Find the matching function in the data
    for func in data['flow_functions']:
        if str(func['identifier']) == identifier:
            function_name = func['name']
            function_identifier = func['identifier']
            script_file_name = func['script_path'].split('/')[-1]  # Extracts file name from path
            return_codes = [code['code'] for code in func['return_codes']]  # Extracts all return codes
            return_codes.insert(0, "Please Select This First")
            return function_name, function_identifier, gr.Dropdown.update(script_file_name), gr.Dropdown.update(choices=return_codes), gr.Dropdown.update(value=return_codes[0])
        
    return None, None, None, None, None

def update_return_information(preset_function, selected_return):
    with open(presetPath, 'r') as f:
        data = json.load(f)
        if selected_return is None:
            # Handle the None case, maybe return some default values or error message
            return None, None, None, None, None, None
        identifier = preset_function.split(': ')[0]
    for func in data['flow_functions']:
        if str(func['identifier']) == identifier:
            for code in func['return_codes']:
                if code['code'] == selected_return:
                    # Extract all needed values from the return code
                    mapped_to_instruction = code.get('mapped_to_instruction', None)
                    willGenerate = code.get('willGenerate', None)
                    saveInputToHistory = code.get('saveInputToHistory', None)
                    saveResponseToHistory = code.get('saveResponseToHistory', None)
                    useContext = code.get('useContext', None)

                    return code['code'], mapped_to_instruction, willGenerate, saveInputToHistory, saveResponseToHistory, useContext

    # If no matching function or return code is found
    return None, None, None, None, None, None
        

def list_flow_presets():
    preset_path = "extensions/cognition_flow/flow_presets"
    return [f for f in os.listdir(preset_path) if f.endswith('.json')]

def list_preset_functions():
    with open(presetPath, 'r') as f:
        data = json.load(f)
    
    return [(str(func['identifier']) + ': ' + func['name']) for func in data['flow_functions']]

def list_flow_functions():
    preset_path = "extensions/cognition_flow/flow_functions"
    return [f for f in os.listdir(preset_path) if f.endswith('.py')]

def create_new_preset(new_preset_name):
    # Logic to create a new JSON file for the preset with the given name
    new_preset_path = f"extensions/cognition_flow/flow_presets/{new_preset_name}.json"
    
    # Check if the file already exists to avoid overwriting
    if os.path.exists(new_preset_path):
        # Handle the case where the file already exists, possibly by returning an error message or asking for a different name
        return "Preset name already exists. Choose a different name."
    scriptPlaceholder = list_flow_functions()
    
    if not scriptPlaceholder:
        return "No flow functions found. Add a script to the flow_functions directory first."
    # Default structure for a new preset JSON file
    new_preset_data = {
        "preset_name": new_preset_name,
        "max_gen_num": 5,
        "max_loop_num": 5,
        "flow_functions": [{
            "name": "Begin",
            "identifier": 0,
            "script_path": f"extensions/cognition_flow/flow_functions/{scriptPlaceholder[0]}",
            "return_codes": [{
                "code": 1,
                "mapped_to_instruction": 1,
                "willGenerate": True,
                "saveInputToHistory": True,
                "saveResponseToHistory": True,
                "useContext": True
            }]
        }]
    }
    
    # Write the new preset data to a JSON file
    with open(new_preset_path, 'w') as f:
        json.dump(new_preset_data, f)
    global preset_dropdown
    # Update the preset dropdown choices
    preset_choices = list_flow_presets()
    
    # Set the dropdown to the newly created preset
    preset_dropdown_value = f"{new_preset_name}.json"
    refresh_ui()
    return gr.Dropdown.update(choices=preset_choices), preset_dropdown_value, ""  # Return the new preset name to update the dropdown in the UI

def refresh_ui():
    global preset_dropdown
    preset_choices = list_flow_presets()
    preset_dropdown.update(choices=preset_choices)
    return gr.Dropdown.update(choices=preset_choices)

def refreshFunctions():
    return gr.Dropdown.update(choices=list_flow_functions())






