import json
from gpt_batch.batcher import GPTBatcher

def read_templates(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        templates = json.load(file)
    return templates

def construct_message_list(templates):
    message_list = []
    for image_id, template in templates.items():
        for obj in template['objects']:
            context_sentences = template['captions']
            context_prompt = f"""
           Task: I will provide a central object. Please identify the environmental context related to this object by focusing exclusively on its surrounding elements. Avoid using phrases that suggest the object is the subject or directly connected to the environment (e.g., "with"). Instead, emphasize its spatial or contextual surroundings using prepositions (e.g., near, on, in).
                Input:
                {obj}
                Context sentences:
                {context_sentences}
                Instruction: Extract only the environmental keywords or phrases (e.g., snow, slope), using prepositions to describe the object's surroundings. Do not include references to other objects (e.g., person, dog).
                Output Format:
                {obj}: {{preposition + environmental keywords}}
            """
            complex_context_prompt = f"""
            Task: I will provide a central object. Please extract the context related to this object and generate a refined description summarizing the environmental details in a clear and concise sentence.
            Input: Obj: {obj}
            Context sentences: {context_sentences}
            Instruction: Make the obj as a center word, to make a sentence with as much details based on the context sentences. Refine the context to describe the environment and setting more succinctly while keeping it specific and relevant to the object.
            Start with the object and use the context sentences to create a detailed and concise description of the environment.
            Output Format: 
            """
            hallucination_prompt = f"""
            Task: I will provide a central object. Please extract the context related to this object and create a more imaginative description. Add plausible details or elements that enhance the setting.
            Input: Obj: {obj}
            Context sentences: {context_sentences}
            Instruction: Refine the context and enrich it with imaginative details to create a vivid and engaging scene. Add a lot of not realated context.
            Output Format: 
            Direltly give the output in the format: {obj} with {{refined and imaginative description}} with 30 words.
            Don't give any othter thing.
            """
            message_list.append({
                "context_prompt": context_prompt,
                "complex_context_prompt": complex_context_prompt,
                "hallucination_prompt": hallucination_prompt
            })
    return message_list

def update_templates_with_results(templates, results):
    result_index = 0
    for image_id, template in templates.items():
        for obj in template['objects']:
            result = results[result_index]
            template['prompts']['context'][obj] = result['context']
            template['prompts']['complex_context'][obj] = result['complex_context']
            template['prompts']['hallucination'][obj] = result['hallucination']
            result_index += 1
    return templates

def save_templates(file_path, templates):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(templates, file, ensure_ascii=False, indent=4)

def main():
    input_file = 'PROMPTS_TEMPLATE.JSON'
    output_file = 'Updated_PROMPTS_TEMPLATE.JSON'
        # Add at top of modified_prompt.py
    from dotenv import load_dotenv
    import os
    
    # Update main() function
    load_dotenv()
        
    model_name = 'gpt-3.5-turbo-1106'
    api_key = os.getenv('api_key')  # Get API key from .env

    # Read templates
    templates = read_templates(input_file)
    # Construct message list
    message_list = construct_message_list(templates)

    # Separate prompts into lists
    context_prompts = [msg['context_prompt'] for msg in message_list]
    complex_context_prompts = [msg['complex_context_prompt'] for msg in message_list]
    hallucination_prompts = [msg['hallucination_prompt'] for msg in message_list]

    # Initialize the batcher
    batcher = GPTBatcher(api_key=api_key, model_name=model_name)

    # Batch process all prompts
    context_results = batcher.handle_message_list(context_prompts)
    complex_context_results = batcher.handle_message_list(complex_context_prompts)
    hallucination_results = batcher.handle_message_list(hallucination_prompts)

    # Combine results
    results = [
        {
            "context": ctx,
            "complex_context": comp_ctx,
            "hallucination": hall
        }
        for ctx, comp_ctx, hall in zip(context_results, complex_context_results, hallucination_results)
    ]

    # Update templates with results
    updated_templates = update_templates_with_results(templates, results)

    # Save updated templates
    save_templates(output_file, updated_templates)

    print(f"Updated templates saved to {output_file}")

if __name__ == "__main__":
    main()