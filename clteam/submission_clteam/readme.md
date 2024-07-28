# CLTeam
## Members

## Systems
All our systems work with graphs extracted from the dialogues. 
We first prompt GPT-4o to extract entities and relationships from the english data of the train and validation sets, and bilingually for the test set. Then we use spacy's NeuralCoref for co-reference, limiting the nodes to a maximum of 100.

### T5GenerationWithGraph
We build a unified framework (T5GenerationWithGraph) that uses graph and text input in a single architecture, where our model learns to generate target sequences in text based on the dialogue history in graph form and the source sequence. For this we fine-tune 'declare-lab/flan-alpaca-base' on the provided training data as well as the extracted graphs.

#### Full History [Primary]

#### Previous Utterance / ... [Constrastive]


### Towerblocks [Constrastive]
Using TowerInstruct-7B-v0.2 we prompt the model with both the dialogue history in form of textual triples and the source sequence to generate the translated target sequence.

