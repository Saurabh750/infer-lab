from engine import InferenceEngine

def main():
    prompts = [
                'What is Quantum Mechanics?', 
                'Who is Albert Einstein?', 
                'Who is President of India?',
                'Which is the best trek in Europe?',
                'What is the national animal of Russia?' 
            ]
    
    engine = InferenceEngine('openai-community/gpt2')

    # for prompt in prompts:
    print(engine.generate(prompts, max_new_tokens=25))

if __name__ == '__main__':
    main()