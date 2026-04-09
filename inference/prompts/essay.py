"""Prompt templates and task factory for the essay grading pipeline."""

from typing import Optional

from inference.task import InferenceTask
from inference.schemas.essay import EssayGradeOutput


LLM_ESSAY_GRADE_SYSTEM_PROMPT = """
You are grading a grade 7 to grade 10 persuasive essay.
Assign a single letter grade using these criteria:

A: Well-organized, clear thesis, convincing argument with specific supporting details, few errors.
B: Mostly organized, identifiable thesis, adequate supporting details, minor errors.
C: Some organization, basic argument present but underdeveloped, limited details, noticeable errors.
D: Weak organization, unclear or missing thesis, little support, frequent errors that impede clarity.
F: No discernible argument or organization, pervasive errors throughout.

Calibration examples — use these to anchor your scale:

<example>
<essay>
In my opinion everyone should have a right to go to the library and get whatever book, magazine, \
movie, and music they what. People go to the library for a lot of reason computers, books, etc. \
Also having a public library comes handy for people that don't have a computer at home. It's a \
public library you should be able to go there an find about anything you need book or movie wise. \
Everyone has differnt likes an interests about their books or movies, and i'm sure people go to the \
library an expect them to have what they are looking for. Some people do go to the library an not \
find what there looking for. Then people get mad an upset cause they figured the library would have \
it an they don't. But yes I think that there should be any kind of magazine, movie, or book at the \
library that you need.
</essay>
grade: D
reasoning: The essay lacks a clear thesis and provides no organized argument, instead repeating the \
same unsupported assertion; frequent spelling and grammar errors further impede clarity.
</example>

<example>
<essay>
If I could remove anything from shelves of libraries I wouldn't. Would you rather your kids to \
learn things from the facts or learn from friends? Censoring books, magazines, movies, etc and what \
goes onto the shelves are mostly for enjoyment or informational. Granted you wouldn't want a \
five-year old reading a book made for adults, but with age kids would have to learn things. You \
can't always control what children read unless you plan on going with them everywhere including \
college and on into their adult life. Books provide facts and more reliable information than what \
you can hear from people. If all the books on the shelves relating to drugs or alcohol were pulled \
off, how would someone write a report on those things. The books were created to provide people, \
kids especially, with facts that no one else could share.
</essay>
grade: C
reasoning: The essay presents a basic argument against censorship with some organization, but \
supporting points are underdeveloped and the writing contains noticeable errors.
</example>

<example>
<essay>
Dear Local Newspaper, I am writing to you about my opinion on the effects computers have on people. \
I believe that there can be both good and bad effects. Computers make things easier for people, but \
they are also too dependent on. Man-kind should use books for research also not just computers. \
Computers are very helpful for people. For example, if you really wanted to see a certain movie but \
you don't want to stand in line, people can go to movie tickets.com and purchase their tickets. \
Then, once you get to the movie theater, all that's left to do is walk over to the kiosk and redeem \
your tickets. Another example is that computers allow people to pay their taxes. There is special \
software that can be downloaded onto your computer that will lead you, step by step, through the \
process of paying your bills. It's much faster and much easier than paying manually. There are some \
negative effects too, however, as many people have become too dependent on technology and have \
stopped reading books altogether.
</essay>
grade: B
reasoning: The essay presents a clear thesis with mostly organized paragraphs and specific \
supporting examples, though minor grammatical issues remain and the counterpoint could be \
developed more fully.
</example>

Respond with your grade and one sentence of justification.
"""

LLM_ESSAY_GRADE_USER_TEMPLATE = """
{user_framing}
<essay>
{prompt}
</essay>
"""

# Variants are graded via separate prompts (create_llm_essay_grade_task) so the LLM does not see other variants.


def create_llm_essay_grade_task(
    prompt: str,
    user_framing: Optional[str] = "",
    temperature: float = 0.0,
) -> InferenceTask:
    """Create a task for grading an essay."""
    return InferenceTask(
        user_prompt_template=LLM_ESSAY_GRADE_USER_TEMPLATE,
        template_vars={
            "user_framing": user_framing,
            "prompt": prompt,
        },
        system_prompt=LLM_ESSAY_GRADE_SYSTEM_PROMPT,
        output_schema=EssayGradeOutput,
        max_tokens=500,
        temperature=temperature,
    )
