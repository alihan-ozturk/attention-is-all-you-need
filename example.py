import torch

batch = 8
numWordtoPred = 10

transformerModel = torch.nn.Transformer(nhead=2, num_encoder_layers=3, d_model=32)
src = torch.rand((1, batch, 32)) # input sentence
tgt = torch.zeros((1, batch, 32)) # <SOS>
output = transformerModel(src, tgt)

outputWords = torch.zeros((batch, numWordtoPred), dtype=torch.long)

for i in range(numWordtoPred):
    output = transformerModel(src, output)
    lastPredWords = output.argmax(dim=2)
    outputWords[:, i] = lastPredWords
    src = torch.cat((src, output), dim=0)
    
print(outputWords)
