import random
random.seed(0)
from slm import *

def get_simple_model(Tokenizer=SpaceTokenizer, Predictor=FrequencyTablePredictor):
    context_length = 2
    tokenizer = Tokenizer()
    predictor = Predictor(context_length)
    model = LanguageModel(
            tokenizer=tokenizer,
            predictor=predictor
            )
    return model

def test_basics():
    np.random.seed(0)
    random.seed(0)
    text = "foo bar foo baz"
    
    model = get_simple_model()
    tokens = model.tokenizer(text)
    
    assert tokens == ['foo', 'bar', 'foo', 'baz']
    # NOTE: Doesn't work for some stupid tokenizer like WhitespaceTokenizer
    assert model.tokenizer.decode(tokens) == text
    
    model.train(tokens)

    initial_context = tokens[:model.context_length]
    generator = model.generate(initial_context)
    output_tokens = list(generator)
    output_text = model.tokenizer.decode(output_tokens)
    assert output_text == "foo bar foo baz"

def test_freq_table_bailout():
    random.seed(0)
    np.random.seed(0)
    text = "foo bar foo baz foo"
    
    context_length = 2
    tokenizer = Gpt2Tokenizer()
    vocabulary_size = len(tokenizer.tokenizer)

    predictor = FrequencyTablePredictor(context_length, bail_to_random=True)
    
    model = LanguageModel(
            tokenizer=tokenizer,
            predictor=predictor
            )

    tokens = model.tokenizer(text)
    
    # NOTE: Doesn't work for some stupid tokenizer like WhitespaceTokenizer
    assert model.tokenizer.decode(tokens) == text
    
    model.train(tokens)
    initial_context = tokenizer("This is not in corpus")
    #model.predictor.sample_most_likely = True
    generator = model.generate(initial_context, max_tokens=10)
    output_tokens = list(generator)
    output_text = model.tokenizer.decode(output_tokens)
    assert output_text == "This is not in corpusazaz bar baz fooaz baz foo"

if __name__ == "__main__":
    #test_fast_table()
    #test_fast_embedding_table()
    test_freq_table_bailout()

