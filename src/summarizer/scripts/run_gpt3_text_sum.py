import os
import logging
from summarizer.model.gpt_summarizer import Gpt3Summarizer, count_tokens
from summarizer.util import SRC_RESOURCES_DIR

MODE_NAME = "text-davinci-003"
TEMPERATURE = 0.7
MAX_TOKENS = 600
TOKEN_RATIO = 0.9
TOP_P = 1.0
FREQUENCY_PENALTY = 0.0
PRESENCE_PENALTY = 1


def poc_sum():
    output_format = "\n\nUse the following format:\n\nIntro: [intro text]\n\nSummary: [summary text]\n\nConclusion: [conclusion text]."

    import os
    import openai

    TLDR_TAG = "\n\nTl;dr"

    text = "Considering the difficulties which men have had to hold to a newly\nacquired state, some might wonder how, seeing that Alexander the\nGreat became the master of Asia in a few years, and died whilst it\nwas scarcely settled (whence it might appear reasonable that the whole\nempire would have rebelled), nevertheless his successors maintained\nthemselves, and had to meet no other difficulty than that which arose\namong themselves from their own ambitions.\n\nI answer that the principalities of which one has record are found to\nbe governed in two different ways; either by a prince, with a body\nof servants, who assist him to govern the kingdom as ministers by his\nfavour and permission; or by a prince and barons, who hold that dignity\nby antiquity of blood and not by the grace of the prince. Such barons\nhave states and their own subjects, who recognize them as lords and hold\nthem in natural affection. Those states that are governed by a prince\nand his servants hold their prince in more consideration, because in all\nthe country there is no one who is recognized as superior to him, and\nif they yield obedience to another they do it as to a minister and\nofficial, and they do not bear him any particular affection.\n\nThe examples of these two governments in our time are the Turk and the\nKing of France. The entire monarchy of the Turk is governed by one lord,\nthe others are his servants; and, dividing his kingdom into sanjaks, he\nsends there different administrators, and shifts and changes them as\nhe chooses. But the King of France is placed in the midst of an ancient\nbody of lords, acknowledged by their own subjects, and beloved by them;\nthey have their own prerogatives, nor can the king take these away\nexcept at his peril. Therefore, he who considers both of these states\nwill recognize great difficulties in seizing the state of the Turk,\nbut, once it is conquered, great ease in holding it. The causes of the\ndifficulties in seizing the kingdom of the Turk are that the usurper\ncannot be called in by the princes of the kingdom, nor can he hope to be\nassisted in his designs by the revolt of those whom the lord has around\nhim. This arises from the reasons given above; for his ministers, being\nall slaves and bondmen, can only be corrupted with great difficulty, and\none can expect little advantage from them when they have been corrupted,\nas they cannot carry the people with them, for the reasons assigned.\nHence, he who attacks the Turk must bear in mind that he will find him\nunited, and he will have to rely more on his own strength than on the\nrevolt of others; but, if once the Turk has been conquered, and routed\nin the field in such a way that he cannot replace his armies, there\nis nothing to fear but the family of this prince, and, this being\nexterminated, there remains no one to fear, the others having no credit\nwith the people; and as the conqueror did not rely on them before his\nvictory, so he ought not to fear them after it.\n\nThe contrary happens in kingdoms governed like that of France, because\none can easily enter there by gaining over some baron of the kingdom,\nfor one always finds malcontents and such as desire a change. Such men,\nfor the reasons given, can open the way into the state and render the\nvictory easy; but if you wish to hold it afterwards, you meet with\ninfinite difficulties, both from those who have assisted you and from\nthose you have crushed. Nor is it enough for you to have exterminated\nthe family of the prince, because the lords that remain make themselves\nthe heads of fresh movements against you, and as you are unable either\nto satisfy or exterminate them, that state is lost whenever time brings\nthe opportunity.\n\nNow if you will consider what was the nature of the government of\nDarius, you will find it similar to the kingdom of the Turk, and\ntherefore it was only necessary for Alexander, first to overthrow him in\nthe field, and then to take the country from him. After which victory,\nDarius being killed, the state remained secure to Alexander, for the\nabove reasons. And if his successors had been united they would have\nenjoyed it securely and at their ease, for there were no tumults raised\nin the kingdom except those they provoked themselves.\n\nBut it is impossible to hold with such tranquillity states constituted\nlike that of France. Hence arose those frequent rebellions against the\nRomans in Spain, France, and Greece, owing to the many principalities\nthere were in these states, of which, as long as the memory of them\nendured, the Romans always held an insecure possession; but with the\npower and long continuance of the empire the memory of them passed\naway, and the Romans then became secure possessors. And when fighting\nafterwards amongst themselves, each one was able to attach to himself\nhis own parts of the country, according to the authority he had assumed\nthere; and the family of the former lord being exterminated, none other\nthan the Romans were acknowledged.\n\nWhen these things are remembered no one will marvel at the ease with\nwhich Alexander held the Empire of Asia, or at the difficulties which\nothers have had to keep an acquisition, such as Pyrrhus and many more;\nthis is not occasioned by the little or abundance of ability in the\nconqueror, but by the want of uniformity in the subject state.\n\n\n\n"

    pp = f"Please summarize the following text:\n{text}" + output_format
    print(count_tokens(text))
    response = openai.ChatCompletion.create(
        model="gpt-4",  # The name of the OpenAI chatbot model to use
        messages=[
            {"role": "system", "content": pp}
        ],   # The conversation history up to this point, as a list of dictionaries
        max_tokens=380,        # The maximum number of tokens (words or subwords) in the generated response
        stop=None,              # The stopping sequence for the generated response, if any (not used here)
        temperature=0.7,        # The "creativity" of the generated response (higher temperature = more creative)
    )
    print(response)
    print(response["choices"][0]["message"]["content"])


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    poc_sum()
    # text_tokens = int(TOKEN_RATIO * MAX_TOKENS)
    # summary_tokens = MAX_TOKENS - text_tokens
    # gpt3_text_sum = Gpt3Summarizer(
    #     model_name=MODE_NAME,
    #     temperature=TEMPERATURE,
    #     text_tokens=text_tokens,
    #     summary_tokens=summary_tokens,
    #     top_p=TOP_P,
    #     frequency_penalty=FREQUENCY_PENALTY,
    #     presence_penalty=PRESENCE_PENALTY
    # )
    #
    # file_path = os.path.join(SRC_RESOURCES_DIR, "sample-text.txt")
    # with open(file_path) as fp:
    #     text = fp.read()
    #     logging.info(gpt3_text_sum.summarize(text))
    #
    # print("hello")
