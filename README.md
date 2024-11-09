## Model :

The model used is a variation of the bert-uncased model that is dynamically fine tuned using the SQuAD dataset.  The model generates answers to questions based on a given context.

## Dataset :

The model is trained on the SQuAD dataset with simulated spoken language modifications and errors. There are several versions of the dataset to simulate varying levels of noise, qualified as a WER (Word Error Rate).  The goal of the dataset is to simulate spoken language errors.

Spoken-SQuAD data files: [Spoken-SQuAD Dataset Repository](https://github.com/chiahsuan156/Spoken-SQuAD)

## Evaluation :

The model is evaluated on three versions of the Spoken-SQuAD dataset, reflecting different WER levels to simulate various noise conditions in spoken language:

- No noise (22.73% WER)
- Noise V1 (44.22% WER)
- Noise V2 (54.82% WER)

### Results :

- **Test Set (NO NOISE - 22.73% WER)** - Exact Match: 64.23%, F1 Score: 74.29%
- **Test V1 Set (V1 NOISE - 44.22% WER)** - Exact Match: 40.72%, F1 Score: 54.94%
- **Test V2 Set (V2 NOISE - 54.82% WER)** - Exact Match: 28.50%, F1 Score: 41.41%