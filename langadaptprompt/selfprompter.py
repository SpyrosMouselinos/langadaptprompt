SELFPROMPT_SYSTEM_PROMPT = "You are a helpful assistant who is an expert on Euclidean geometry. " \
                           "Your will be given a task and a list of solved problems. " \
                           "Imagine you need to give an answer for the task, which of the problems would be useful hints? " \
                           "Return only the indexes of the most relevant solved problems in a list (e.g [1,5,4]). Do not return anything else."

SELFPROMPT_USER_PROMPT = (
    "Unsolved Problem:\nDescription: Construct an angle of 30° with the given side. Let A be the initial point of the given ray.\n"
    "Solved Problems:\n"
    "1)\nDescription: Construct an angle of 60° with the given side.Let A be the initial point of the given ray.\n"
    "2)\nDescription: Inscribe a circle in the square.Given the square ABCD.\n"
    "3)\nDescription: Construct a line through the given point that cuts the rectangle into two parts of equal area.Given the rectangle ABCD and the point O.\n"
    "4)\nDescription: Let |AB|=1. Construct a point C on the ray AB such that the length of AC is equal to √2.\n"
    "5)\nDescription: Reflect the segment across the line.Given the segment AB.\n"
    "6)\nDescription: Construct a circle that passes through the midpoints of sides of the given acute triangle.Given the triangle ABC.")
SELFPROMPT_ASSISTANT_PROMPT = "Most Relevant Problems are: [1,3]"

SELFPROMPT_USER_PROMPT2 = (
    "Unsolved Problem:\nDescription: Inscribe a circle in the rhombus. Given the rhombus ABCD.\n"
    "Solved Problems:\n"
    "1)\nDescription: Construct an angle of 60° with the given side.Let A be the initial point of the given ray.\n"
    "2)\nDescription: Inscribe a circle in the square.Given the square ABCD.\n"
    "3)\nDescription: Construct a line through the given point that cuts the rectangle into two parts of equal area.Given the rectangle ABCD and the point O.\n"
    "4)\nDescription: Let |AB|=1. Construct a point C on the ray AB such that the length of AC is equal to √2.\n"
    "5)\nDescription: Reflect the segment across the line.Given the segment AB.")
SELFPROMPT_ASSISTANT_PROMPT2 = "Most Relevant Problems are: [2,4]"