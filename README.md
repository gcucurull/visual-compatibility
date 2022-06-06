
# Context-Aware Visual Compatibility Prediction

## Requirements

The model is implemented with Tensorflow. All necessary libraries can be installed with:

    pip install -r requirements.txt

## Data

### Polyvore
The [Polyvore dataset](https://github.com/xthan/polyvore-dataset) can be automatically downloaded by running the following script in `data/`:

    ./get_polyvore.sh
 
    python train.py -d polyvore

## License
`MIT`
