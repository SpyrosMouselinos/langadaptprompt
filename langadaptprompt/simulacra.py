import random
import openai
import torch
import gc
import time
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from .promptbuilder import few_shot_builder
from .constants import get_nl_s_prompt, get_nl_v_prompt, get_gt_s_prompt, get_gt_v_prompt
from tqdm import tqdm


def extract_descriptions_and_solutions(text):
    # Splitting the text into segments
    segments = text.split("Description:")[1:]  # Skipping the first split as it will be empty
    pairs = []

    for segment in segments:
        # Further split each segment into description and solution
        description, solution = segment.split("Solution:")
        # Cleaning up and formatting the segments
        description = "Description: " + description.strip()
        solution = "Solution: " + solution.strip().split("Description:")[
            0]  # To handle cases where another Description follows
        # Adding the pair to the list
        pairs.append([description])
        pairs.append([solution])
    pairs[-2] = [pairs[-2][0] + '\n' + pairs[-1][0]]
    pairs = pairs[:-1]
    return pairs


def cappend(item, lista):
    if isinstance(lista, list):
        lista.append(item)
        return lista
    else:
        return [lista, item]


def find_difference(new, input_txt):
    min_length = min(len(input_txt), len(new))
    i = 0
    while i < min_length and input_txt[i] == new[i]:
        i += 1
    return new[i:]


def stop_at_specific_tokens(decoded_tokens, stop_tokens):
    stop_positions = []
    for stop_token in stop_tokens:
        if stop_token in decoded_tokens:
            stop_positions.append(decoded_tokens.find(stop_token))
    cutoff = min(stop_positions) if len(stop_positions) > 0 else len(decoded_tokens) + 1
    return decoded_tokens[:cutoff]


class Agent:
    def __init__(self, id, llm, inception_prompt, temp=0.6, openai_key=None):
        self.id = id
        self.llm = llm
        self.inception_prompt = inception_prompt
        self.temp = temp
        self.openai_key = openai_key

    def run_engine(self, message):
        if self.llm == 'fake' and self.id == 1:
            return self._call_fake_s(message)
        if self.llm == 'fake' and self.id == 2:
            return self._call_fake_v(message)

        if isinstance(self.llm, list):
            response = self._call_huggingface_model(message)
        elif isinstance(self.llm, str):
            if 'gpt' in self.llm:
                response = self._call_openai_model(message)
            else:
                raise ValueError
        else:
            response = self._call_vertexai_model(message)
        return response

    def think(self, context, memory=None, fewshot=False):
        if fewshot and 'gpt' in self.llm:
            if isinstance(context, str):
                memory = extract_descriptions_and_solutions(context)
            elif isinstance(context, list):
                memory = extract_descriptions_and_solutions(context[0])
                for i in range(1, len(context)):
                    memory.append(context[i])
            return self.run_engine(memory), memory[-1][0]
        if isinstance(context, str):
            original = extract_descriptions_and_solutions(context)[-1][0]
            memory = [context]
        else:
            memory = context
            original = None
        return self.run_engine(memory), original

    def gather(self, memory, command):
        new_memory = [command + '\n\nConversation:' + '\n'.join(memory)]
        return self.run_engine(new_memory)

    def validate(self, memory):
        flat_memory = []
        for item in memory:
            if isinstance(item, str):
                flat_memory.append(item)
            elif isinstance(item, list):
                for iitem in item:
                    flat_memory.append(iitem)
        new_memory = ["Are the following solution steps correct?\n\n" + '\n'.join(flat_memory)]
        return self.run_engine(new_memory)

    def shift(self, problem, solution_1, solution_2, command):
        new_memory = [
            command + f'\n\nProblem:\n{problem}' + f'\nSolution 1:{solution_1}\n' + f'\nSolution 2:{solution_2}']
        return self.run_engine(new_memory)

    def _call_fake_s(self, messages):
        # Responds with a predefined step list
        return "Step 1. Do this\nStep 2. Do that"

    def _call_fake_v(self, messages):
        # Randomly selects a response
        responses = ["all is good", "Step 1 is incorrect", "Step 2 is incorrect"]
        return random.choice(responses)

    def _call_openai_model(self, messages):
        split_messages = []
        split_messages.append({"role": "system", "content": self.inception_prompt})
        role_list = ["user", "assistant"]
        for i in range(0, len(messages)):
            if isinstance(messages[i], list):
                fixed_message = messages[i][0]
            else:
                fixed_message = messages[i]
            split_messages.append({"role": role_list[i % 2], "content": fixed_message})
        retries = 10
        while retries > 0:
            try:
                solution = openai.ChatCompletion.create(
                    api_key=self.openai_key,
                    model=self.llm,
                    max_tokens=300,
                    stop=['\n\n\n'],
                    messages=split_messages,
                    temperature=self.temp,
                    top_p=0.95,
                    n=1
                )
                response = [choice['message']['content']
                            for choice in solution['choices']][0]
                return response
            except Exception as e:
                time.sleep(5)
                retries -= 1
        return messages

    def _call_huggingface_model(self, messages):
        llm, tokenizer = self.llm
        augmented_messages = self.inception_prompt + '\n\n' + '\n'.join(messages)
        inputs = tokenizer(augmented_messages, return_tensors="pt")
        # inputs = inputs.to(device='cuda')
        stop_tokens = ["\n\n\n", "Example", "Problem", "Description"]

        with torch.no_grad():
            outputs = llm.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                top_p=0.95,
                temperature=self.temp,
                num_return_sequences=1
            )
        ret = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        cache_out = [stop_at_specific_tokens(find_difference(ret[i], augmented_messages), stop_tokens) for i in
                     range(len(ret))]
        gc.collect()
        torch.cuda.empty_cache()
        return cache_out[0].strip()

    def _call_vertexai_model(self, messages):
        augmented_messages = self.inception_prompt + '\n\n' + '\n'.join(messages)
        retries = 10
        while retries > 0:
            try:
                response = self.llm.generate_content(
                    augmented_messages,
                    generation_config={
                        'temperature': self.temp,
                        'top_p': 0.95,
                    },
                )
                return response
            except Exception as e:
                print("Waiting for vertex...")
                retries -= 1
                time.sleep(5)
        return messages


class Simulacra:
    def __init__(self, agents):
        if len(agents) != 4:
            raise ValueError("Simulacra requires exactly 4 agents")
        self.agents = {agent.id: agent for agent in agents}
        llm = agents[0].llm
        for agent in agents:
            if agent.llm != llm:
                raise ValueError("All agents must have the same llm")
            if list(self.agents.values()).count(agent) > 1:
                raise ValueError("All agents must have different ids")

        self.gather_agent = Agent(id=5, llm=self.agents[2].llm,
                                  inception_prompt="You are an expert in reading and analyzing text."
                                                   "You are provided with a dialogue where solution steps of"
                                                   " a problem are suggested and then corrected. "
                                                   "Corrections refer to only specific steps. "
                                                   "Your goal is to read the dialogue below, "
                                                   "apply the corrections to the initial set "
                                                   "of steps and and return all the steps "
                                                   "corrected. List the final, correct steps in a clear,"
                                                   " bullet-point format"
                                  )
        self.shift_agent = Agent(id=6, llm=self.agents[2].llm,
                                 inception_prompt="You are an expert in euclidean geometry, logic and mathematics. "
                                                  "You are provided with a problem description and "
                                                  "two solutions for it. "
                                                  "Your goal is to assess which solution is better. "
                                                  "If both solutions seem equally good then prefer the first solution. "
                                 )
        self.cache_original_nl_problem = None
        self.cache_original_fs_nl = None
        self.cache_original_nl_solution = None
        self.cache_original_gt_problem = None
        self.cache_original_fs_gt = None
        self.cache_original_gt_solution = None

    def nl_solve(self, messages):
        max_iterations = 2
        agent1 = self.agents[1]
        agent2 = self.agents[2]

        for iteration in range(max_iterations):
            if iteration == 0:
                # NL Solver thinks about the problem
                self.cache_original_fs_nl = messages
                response1, original_problem = agent1.think(messages, fewshot=True)
                self.cache_original_nl_problem = original_problem.split('Solution:')[0].strip() + '\nSolution:\n'
                self.cache_original_nl_solution = response1
                messages = response1
            else:
                # NL Solver thinks about the problem
                response1, _ = agent1.think([self.cache_original_fs_nl] + messages, fewshot=True)
                messages = cappend(response1, messages)

            # NL Validator validates the steps
            response2 = agent2.validate([self.cache_original_nl_problem, messages])
            messages = cappend(response2, messages)

            # Check if NL Validators' response indicates validation
            if "yes" in response2.lower() or ("all" in response2.lower() and (
                    "correct" in response2.lower() or "valid" in response2.lower())) or (
                    "steps are correct" in response2.lower()) or ("steps are valid" in response2.lower()):
                # Extract and return the final steps from the messages
                return self._extract_steps(messages)

            # Check for specific errors mentioned by NL Validator
            elif "no" in response2.lower() or "incorrect" in response2.lower() or "not correct" in response2.lower() or "isnt correct" in response2.lower() or "isn't correct" in response2.lower() or "invalid" in response2.lower() or "not valid" in response2.lower() or "isnt valid" in response2.lower() or "isn't valid" in response2.lower():
                continue  # Continue the loop for another iteration

        # Return the final steps after max iterations
        return self._extract_steps(messages)

    def gt_solve(self, messages):
        max_iterations = 2
        agent3 = self.agents[3]
        agent4 = self.agents[4]

        for iteration in range(max_iterations):
            if iteration == 0:
                # GT Solver thinks about the problem
                self.cache_original_fs_gt = messages
                response1, original_problem = agent3.think(messages, fewshot=True)
                self.cache_original_gt_problem = original_problem.split('Solution:')[0].strip() + '\nSolution:\n'
                self.cache_original_gt_solution = response1
                messages = response1
            else:
                # GT Solver thinks about the problem
                response1, _ = agent3.think([self.cache_original_fs_gt] + messages, fewshot=True)
                messages = cappend(response1, messages)

            # GT Validator validates the steps
            response2 = agent4.validate([self.cache_original_gt_problem, messages])
            messages = cappend(response2, messages)

            # Check if NL Validators' response indicates validation
            if ("all" in response2.lower() and ("correct" in response2.lower() or "valid" in response2.lower())) or (
                    "steps are correct" in response2.lower()) or ("steps are valid" in response2.lower()):
                # Extract and return the final steps from the messages
                return self._extract_steps(messages, 'gt')

            # Check for specific errors mentioned by NL Validator
            elif "incorrect" in response2.lower() or "not correct" in response2.lower() or "isnt correct" in response2.lower() or "isn't correct" in response2.lower() or "invalid" in response2.lower() or "not valid" in response2.lower() or "isnt valid" in response2.lower() or "isn't valid" in response2.lower():
                continue  # Continue the loop for another iteration

        # Return the final steps after max iterations
        return self._extract_steps(messages, 'gt')

    def domain_shift(self, nl_solution):
        shift_agent = self.shift_agent
        if self.cache_original_nl_problem is None:
            raise ValueError("You must first run the NL agents!")
        shift_command = (
            "Reply with \"1\" if you prefer the first solution or \"2\" if you prefer the second."
            "Do not reply with anything else."
        )
        result = shift_agent.shift(problem=self.cache_original_nl_problem, solution_1=nl_solution,
                                   solution_2=self.cache_original_nl_solution, command=shift_command)

        try:
            sol = int(result.strip())
            if sol == 1:
                return nl_solution
            else:
                return self.cache_original_nl_solution
        except ValueError:
            if '1' in result:
                return nl_solution
            elif '2' in result:
                return self.cache_original_nl_solution
            else:
                return self.cache_original_nl_solution

    def _extract_steps(self, messages, mode='nl'):
        gather_agent = self.gather_agent
        if mode == 'nl':
            extraction_command = (
                "Each step should start with 'STEP' followed by the step number and its description. "
                "Do not return more steps than those provided."
                "For example:\n'STEP 1. Draw a Line...\nSTEP 2. Using center X and radius...' and so on."
            )
        elif mode == 'gt':
            extraction_command = (
                "Each step should start with the step number followed by the tool used and its description."
                " Do not return more steps than those provided."
                "For example:\n'1. Line Tool Draw a line...\n2. Circle Tool Using center X and radius...' and so on."
            )
        else:
            extraction_command = ''
        final_steps = gather_agent.gather(messages, extraction_command)
        return final_steps


def run_simulacra(llm, key=None,
                  adapt_self=False,
                  top_k=1,
                  solver_temp=0.2,
                  validator_temp=0.8,
                  data=None,
                  debug=False):
    print(
        "NOTICE: Argument LLM must be one of [chatgpt3.5, gpt4] for OPENAI models. You must also provide the Key argument."
        "\nAlternatively, it must be the name of a huggingface model currently with context size > 2048."
        "\nOr must be the Gemini Model Instance.")

    if isinstance(llm, str):
        if llm == 'chatgpt3.5':
            llm_ = 'gpt-3.5-turbo-1106'
            assert key is not None
            print("Assuming ChatGPT agents...\n")
        elif llm == 'gpt4':
            llm_ = 'gpt-4-0125-preview'
            assert key is not None
            print("Assuming GPT4 agents...\n")
        else:
            model = AutoModelForCausalLM.from_pretrained(llm)
            tokenizer = AutoTokenizer.from_pretrained(llm)
            llm_ = [model, tokenizer]  # Huggingface
            print(f"Assuming {llm.split('/')[0]} agents...\n")
    else:
        llm_ = llm  # Gemini
        print(f"Assuming Gemini agents...\n")

    a1 = Agent(id=1, llm=llm_,
               inception_prompt=get_nl_s_prompt(), temp=solver_temp, openai_key=key)
    a2 = Agent(id=2, llm=llm_,
               inception_prompt=get_nl_v_prompt(), temp=validator_temp, openai_key=key)
    a3 = Agent(id=3, llm=llm_,
               inception_prompt=get_gt_s_prompt(), temp=solver_temp, openai_key=key)
    a4 = Agent(id=4, llm=llm_,
               inception_prompt=get_gt_v_prompt(), temp=validator_temp, openai_key=key)

    if data is None:
        data = pd.read_csv('new_euclidea.csv')
        data = data[(data['pack'] == 'Alpha') | (data['pack'] == 'Beta')]
    data = data.dropna()
    flat_data = [row.to_dict() for _, row in data.iterrows()]

    results = {'results': []}
    for problem in tqdm(flat_data):
        pack = problem['pack']
        question_nl = problem['question_r'].strip()
        question_gt = problem['question'].strip()
        few_shot_prompt_nl = few_shot_builder(current_question=question_nl,
                                              dataset_df=data,
                                              pack_limit=pack,
                                              mode='Adapt',
                                              self_reflect=adapt_self,
                                              question_kw='question_r',
                                              solution_kw='solution_r',
                                              solution_tool_kw='solution_tool',
                                              )
        few_shot_prompt_gt = few_shot_builder(current_question=question_gt,
                                              dataset_df=data,
                                              pack_limit=pack,
                                              mode='Adapt',
                                              self_reflect=adapt_self,
                                              )
        instance_results = []
        for k in range(top_k):
            if debug:
                try:
                    framework = Simulacra(agents=[a1, a2, a3, a4])
                    few_shot_examples = few_shot_prompt_nl[:few_shot_prompt_nl.rfind('Description:')].strip()
                    nl_solution = framework.nl_solve(
                        messages=f"Here are some solved examples:\n\n{few_shot_examples}"
                                 f"\n\nSolve the following problem:\n\n{question_nl}\n")
                    domain_shift = framework.domain_shift(nl_solution)
                    few_shot_examples = few_shot_prompt_gt[:few_shot_prompt_gt.rfind('Description:')].strip()
                    question_gt = question_gt[:question_gt.rfind('Solution:')].strip()
                    gt_solution = framework.gt_solve(
                        messages=f"Here are some solved examples:\n\n{few_shot_examples}"
                                 f"\n\nSolve the following problem using the available tools.\n\n{question_gt}\n"
                                 f"Here is a proposed solution strategy:\n{domain_shift}\n\nSolution:")
                    instance_results.append([domain_shift, gt_solution])
                except Exception as e:
                    print(f"Exception occurred: {e}\nRetrying...\n")
                    instance_results.append(['', ''])
                    pass
            else:
                framework = Simulacra(agents=[a1, a2, a3, a4])
                few_shot_examples = few_shot_prompt_nl[:few_shot_prompt_nl.rfind('Description:')].strip()
                nl_solution = framework.nl_solve(
                    messages=f"Here are some solved examples:\n\n{few_shot_examples}"
                             f"\n\nSolve the following problem:\n\n{question_nl}\n")
                domain_shift = framework.domain_shift(nl_solution)
                few_shot_examples = few_shot_prompt_gt[:few_shot_prompt_gt.rfind('Description:')].strip()
                question_gt = question_gt[:question_gt.rfind('Solution:')].strip()
                gt_solution = framework.gt_solve(
                    messages=f"Here are some solved examples:\n\n{few_shot_examples}"
                             f"\n\nSolve the following problem using the available tools.\n\n{question_gt}\n"
                             f"Here is a proposed solution strategy:\n{domain_shift}\n\nSolution:")
                instance_results.append([domain_shift, gt_solution])

        results['results'].append(instance_results)
    return results


if __name__ == '__main__':
    print(run_simulacra('chatgpt3.5', key='')) #'Yukang/Llama-2-7b-longlora-32k-ft'

