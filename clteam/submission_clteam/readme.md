# CLTeam

## Members
Lea Krause [Contact person: l.krause@vu.nl]
Selene Báez Santamaría
Jan-Christoph Kalo

## Systems Overview
All our systems work with graphs extracted from dialogues. We employ a multi-step process to extract entities and relationships from the dialogue data and utilize these in various model settings.

### Data Extraction and Preparation
1. **Entity and Relationship Extraction**: 
    - We prompt GPT-4o to extract entities and relationships from the dialogue data.
    - For the train and validation sets, we use only the English data. For the test set, we prompt with bilingual data.
2. **Co-reference Resolution**: 
    - We use spaCy's NeuralCoref to resolve co-references, limiting the number of nodes to a maximum of 100.

### Model Architectures

#### Towerblocks
Using TowerInstruct-7B-v0.2, we prompt the model with both the dialogue history (in the form of textual non-co-referenced triples) and the source sequence to generate the translated target sequence.

##### Variants:
- **Full History [Primary]**: Uses a graph representation of the dialogue history up to the source and the source.
- **No history [Constrastive]**: Uses just the source sequence.


Not submitted:
#### GraphFlanAlpaca
We have developed a unified framework named GraphFlanAlpaca, which integrates both graph and text input into a single architecture. This model is designed to generate target sequences in text based on the dialogue history represented in graph form and the source sequence. Our approach involves:

- **Base Model**: Fine-tuning 'declare-lab/flan-alpaca-base'.
- **Training Data**: Provided training data along with the extracted graphs.

##### Variants:
- **Full History**: Uses a graph representation of the dialogue history and the source.
- **No history**: Uses a graph representation of the source.



