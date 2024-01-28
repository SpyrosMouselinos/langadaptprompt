import os
from crewai import Agent, Task, Crew, Process
from crewai.tools.agent_tools import AgentTools
from langchain.chat_models import ChatOpenAI
import pandas as pd
import json
from typing import List, Optional
from pydantic.v1 import BaseModel, Field, Json, root_validator

os.environ["OPENAI_API_KEY"] = "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"


class AccumulativeCrew(Crew):
    def __init__(self, **data):
        super().__init__(**data)

    def kickoff(self) -> str:
        """
        Kickoff the crew to work on it's tasks.
            Returns:
                output (List[str]): Output of the crew for each task.
        """
        if self.process == Process.sequential:
            return self.__sequential_loop()

    def __sequential_loop(self) -> str:
        """
        Loop that executes the sequential process.
            Returns:
                output (str): Output of the crew.
        """
        task_outcome = None
        for task in self.tasks:
            # Add delegation tools to the task if the agent allows it
            if task.agent.allow_delegation:
                tools = AgentTools(agents=self.agents).tools()
                task.tools += tools

            self.__log(f"\nWorking Agent: {task.agent.role}")
            self.__log(f"Starting Task: {task.description} ...")

            task_outcome += task.execute(task_outcome)

            self.__log(f"Task output: {task_outcome}")

        return task_outcome


class CrewAIProblemSolver:
    def __init__(self, max_rounds):
        self.max_rounds = max_rounds
        self.shared_memory = {'question': '', 'agreed_answers': []}
        self.validator_memory = {'solution_pairs': []}

    def store_final_output(self, output):
        self.shared_memory['final_solution'] = output

    def update_validator_memory(self, initial_solution, final_solution):
        self.validator_memory['solution_pairs'].append((initial_solution, final_solution))

    def conversation_task(self, shared_memory, validator_memory=None):
        agent_a1 = Agent(
            role='Solve Agent',
            goal='Propose a sequence of solution steps for a given problem. '
                 'At the end of your execution return the steps in bullets.',
            backstory="""You are an expert on Euclidean Geometry.
             You will solve geometric problems step by step.""",
            # Here are some problems alongside their solutions:{}""",
            verbose=False,
            allow_delegation=False,
            llm=ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0.5)
        )
        agent_a2 = Agent(
            role='Validation Agent',
            goal='Validate the solution steps for the given problem. '
                 'At the end of your validation return the steps in bullets.',
            backstory="""You are an expert on Euclidean Geometry.
            You will validate solutions to geometric problems step by step.""",
            # Here are some problems alongside their correct solutions:{}""",
            verbose=False,
            allow_delegation=False,
            llm=ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0.8)

        )
        task_solve = Task(
            description=f"{shared_memory['question']}",
            agent=agent_a1)

        task_validate = Task(
            description=f"\n\nNow, go over each solution step and validate its correctness."
                        f"If you believe that is correct, you must repeat it without any changes. "
                        f"If the step is incorrect you must repeat it and add #Incorrect after it."
                        f"Do not comment anything else."
            ,
            agent=agent_a2)

        task_rewrite = Task(
            description=f"\n\nNow, go over the solution steps and if you find incorrect ones, correct them. "
                        f"You must return the corrected steps in bullets. "
                        f"Do not comment anything else."
            ,
            agent=agent_a1)

        crew = AccumulativeCrew(
            agents=[agent_a1, agent_a2],
            tasks=[task_solve, task_validate, task_rewrite, task_validate, task_rewrite],
            process=Process.sequential,
            verbose=False
        )
        out = crew.kickoff()
        print(out)
        return

    def run(self, input_data, problem_col=None, solution_col=None):
        if isinstance(input_data, pd.DataFrame):
            for _, row in input_data.iterrows():
                self.shared_memory['question'] = row[problem_col]
                self.shared_memory['agreed_answers'] = [row[solution_col]]
                self.conversation_task(shared_memory=self.shared_memory)

        elif isinstance(input_data, list):
            for problem, solution in input_data:
                self.shared_memory['question'] = problem
                self.shared_memory['agreed_answers'] = [solution]
                self.conversation_task(shared_memory=self.shared_memory)

        return


data = {
    'problem': ["""
Given two points on a line AB with length equal to 1, construct a point X so that AX = square root of 2.
You can not use numbers in your solution.
AX must be on the same line as AB.
"""],
    'solution': ['Do a backflip']
}
df = pd.DataFrame(data)

solver = CrewAIProblemSolver(1)
result = solver.run(df, 'problem', 'solution')
