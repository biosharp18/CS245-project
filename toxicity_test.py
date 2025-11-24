import evaluate

toxicity = evaluate.load("toxicity", 'DaNLP/da-electra-hatespeech-detection', module_type="measurement",)
toxicity.compute(predictions=["i fucking hate cars", "wow very nice!"], toxic_label='offensive')
breakpoint()