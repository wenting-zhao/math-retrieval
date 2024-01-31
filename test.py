import argparse
import re
from copy import deepcopy
from tqdm import tqdm
from datasets import load_dataset
from utils import chat_compeletion_openai
import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

PROMPT=r"""Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.  The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."""

MORE=r"""Q: Olivia has \$23. She bought five bagels for \$3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", type=str, help="what to retrieve")
    parser.add_argument("--position", type=str, default="end", help="where to put retrieval example")
    parser.add_argument("--long", action="store_true", help="whether to test long context")
    parser.add_argument("--retrieval", action="store_true", help="whether to retrieve")
    args = parser.parse_args()
    ds = load_dataset("minimario/gsm8k-rewritten", split='train')
    acc = 0
    for idx, one in tqdm(enumerate(ds), total=len(ds)):
        if args.retrieval:
            if args.long:
                questions = deepcopy(ds[args.option])
                answers = deepcopy(ds['old_solution'])
                if args.position == "start":
                    questions[0], questions[idx] = questions[idx], questions[0]
                    answers[0], answers[idx] = answers[idx], answers[0]
                elif args.position == "end":
                    questions[-1], questions[idx] = questions[idx], questions[-1]
                    answers[-1], answers[idx] = answers[idx], answers[-1]
                else:
                    questions[len(ds)//2], questions[idx] = questions[idx], questions[len(ds)//2]
                    answers[len(ds)//2], answers[idx] = answers[idx], answers[len(ds)//2]
                context = ""
                for q, a in zip(questions, answers):
                    context += f"Q: {q}\nA: {a}\n\n"
                prompt = f"{PROMPT}\n\n{context}Q: {one['new_question']}\nA: "
            else:
                q = one[args.option]
                a = one['old_solution']
                if args.position == "start":
                    prompt = f"Q: {q}\nA: {a}\n\n{PROMPT}\n\nQ: {one['new_question']}\nA: "
                elif args.position == "end":
                    prompt = f"{PROMPT}\n\nQ: {q}\nA: {a}\n\nQ: {one['new_question']}\nA: "
                else:
                    raise NotImplementedError(f"{args.position} not supported yet.")
        else:
            if args.long:
                context = ""
                questions = deepcopy(ds[args.option])
                answers = deepcopy(ds['old_solution'])
                del questions[idx], answers[idx]
                for q, a in zip(questions, answers):
                    context += f"Q: {q}\nA: {a}\n\n"
                prompt = f"{PROMPT}\n\n{context}{MORE}\n\nQ: {one['new_question']}\nA: "
            else:
                prompt = f"{PROMPT}\n\n{MORE}\n\nQ: {one['new_question']}\nA: "
        text_in = [{"role": "user", "content": prompt}]
        out = chat_compeletion_openai('gpt-3.5-turbo-16k', text_in)
        # assume answer as the last number at the last sentence 
        out = tokenizer.tokenize(out)[-1].replace(',', '')
        answer_float = re.findall("\d+\.\d+", out)
        answer_int = re.findall("\d+", out)
        if len(answer_float) >= 1 and len(answer_int) >= 1:
            if out.rfind(answer_float[-1]) > out.rfind(answer_int[-1]):
                answer = answer_float[-1]
            else:
                answer = answer_int[-1]
        elif len(answer_float) == 0 and len(answer_int) == 0:
            answer = "null"
        elif len(answer_float) == 0:
            answer = answer_int[-1]
        else:
            answer = answer_float[-1]
        if answer == one['new_answer']:
            acc += 1
        print(out, '|', answer, '|', one['new_answer'])
    print(acc/len(ds))

if __name__ == '__main__':
    main()
