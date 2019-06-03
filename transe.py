import logging
import torch
import time
import os

import numpy as np

from os.path import join
from pykeen.utilities import evaluation_utils
from pykeen.kge_models import TransE
from mlflow import log_metric, log_param
import mlflow


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

logging.getLogger('pykeen').setLevel(logging.INFO)


def load_datasets():
    '''Reads in the index mappings and dataset of the FB15k-237 dataset.
    Returns
    -------
        vocab - Dict: The index mapping for the FB15k-237 dataset with the
                      following mappings:
                      e2id --> entity to index
                      id2e --> index to entity
                      id2rel --> index to relation
                      rel2id --> relation to index
        dataset - Dict: A dictionary containing the train, valid and
                        test dataset.   
    '''
    base_path = join('data/processed/FB15k-237/')

    mappings = [
        'entity_to_id.pickle',
        'id_to_entity.pickle',
        'id_to_relation.pickle',
        'relation_to_id.pickle'
    ]

    datasets = [
        'test.pickle',
        'train.pickle',
        'valid.pickle'
    ]

    e2id, id2e, id2rel, rel2id = \
        (np.load(join(base_path, f), allow_pickle=True) for f in mappings)
    test, train, valid = \
        (np.load(join(base_path, f), allow_pickle=True) for f in datasets)

    vocab = dict(
        e2id=e2id,
        id2e=id2e,
        id2rel=id2rel,
        rel2id=rel2id
    )

    datasets = dict(
        train=train,
        valid=valid,
        test=test
    )
    return vocab, datasets

#TODO [1]
def get_model(num_entities, num_relations):
    '''Initializes a ConvE model.
    Parameters
    ----------
        num_entities - int: The total number of distinct entities in the
                            dataset.
        num_relations - int: The total number of distinct realtions in the
                             dataset.
    '''
    model = ConvE(
        margin_loss=1,
        embedding_dim=50,
        scoring_function = 1,
        normalization_of_entities= 2,
        random_seed=1234,
        preferred_device='gpu',
        **kwargs

        # preferred_device='gpu',
        # random_seed=1234,
        # embedding_dim=50,  # Note: embedding_dim = ConvE_width * ConvE_height
        # ConvE_input_channels=1,
        # ConvE_output_channels=3,
        # ConvE_width=10,
        # ConvE_height=5,
        # ConvE_kernel_height=5,
        # ConvE_kernel_width=3,
        # conv_e_input_dropout=0.2,
        # conv_e_feature_map_dropout=0.5,
        # conv_e_output_dropout=0.5,
    )
    
    model.num_entities = num_entities
    model.num_relations = num_relations

    # model._init_embeddings()

    return model


def main():
    num_epochs = 100
    learning_rate = 0.001
    batch_size = 64

    preferred_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(preferred_device)

    vocab, datasets = load_datasets()

    train, test = datasets['train'], datasets['test']

    id2e, id2rel = vocab['id2e'], vocab['id2rel']

    model = get_model(num_entities=len(id2e), num_relations=len(id2rel))

    if os.environ.get('DEV'):
        logger.warn('Running in dev mode')

        num_epochs = 1
        train = train[:100]
        test = test[:20]

    with mlflow.start_run():

        log_param("Epochs", num_epochs)
        log_param("Learning Rate", learning_rate)
        log_param("Batch size", batch_size)

        loss_per_epoch = model.fit(
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            batch_size=batch_size,
            pos_triples=train,
            tqdm_kwargs=None,
        )

        # %% Save the model and the loss
        timestr = time.strftime("%Y%m%d-%H%M%S")

        torch.save(model.state_dict(), f'models/{timestr}-ConvE.pickle')
        torch.save(loss_per_epoch, f'models/{timestr}-ConvE-loss.pickle')

        metrics = evaluation_utils.metrics_computations.compute_metric_results(
            kg_embedding_model=model,
            mapped_train_triples=train,
            mapped_test_triples=test,
            device=device
        )

        log_metric("Mean Rank", metrics.mean_rank)
        log_metric("Mean Reciprocal Rank", metrics.mean_reciprocal_rank)

        for k, v in metrics.hits_at_k.items():
            log_metric(f'Hits_at_{k}', v)


if __name__ == "__main__":
    main()
