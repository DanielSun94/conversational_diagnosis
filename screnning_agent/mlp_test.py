import csv
import math
import os.path
import numpy as np
from read_data import read_data
from eval import top_calculate
from sklearn.neural_network import MLPClassifier
from config import structured_data_folder
from logger import logger
import random
from screnning_agent.config import resource_folder


def data_transform(dataset, fraction, use_data_type):
    diagnosis_list, symptom_list, key_list, embedding_list = [], [], [], []
    for i in range(len(dataset)):
        key, symptom, diagnosis, embedding = dataset.__getitem__(i)
        assert np.sum(np.array(symptom)) > 1
        assert np.sum(np.array(diagnosis)) == 1
        diag_idx = np.argmax(diagnosis)
        diagnosis_list.append(diag_idx)
        symptom_list.append(symptom[2::3])  # 只使用hit这个特征
        key_list.append(key)
        embedding_list.append(embedding)

    if fraction < 1:
        item_list = [i for i in range(len(key_list))]
        random.shuffle(item_list)
        item_list = item_list[: math.ceil(len(item_list) * fraction)]
        new_diagnosis_list = []
        new_symptom_list = []
        new_embedding_list = []
        for item in item_list:
            new_diagnosis_list.append(diagnosis_list[item])
            new_symptom_list.append(symptom_list[item])
            new_embedding_list.append(embedding_list[item])
        diagnosis_list = np.array(new_diagnosis_list)
        symptom_list = np.array(new_symptom_list)
        embedding_list = np.array(new_embedding_list)
    if use_data_type == 0:
        input_list = np.concatenate([symptom_list, embedding_list], axis=1)
    elif use_data_type == 1:
        input_list = symptom_list
    elif use_data_type == 2:
        input_list = embedding_list
    else:
        raise ValueError('')
    return diagnosis_list, input_list, key_list


def main():
    data_path = os.path.join(resource_folder, 'mlp_test_result.csv')
    hidden_layer_sizes = [64, 32]
    max_iter = 500
    faction_list = [1.0, 0.75, 0.5, 0.25]

    model_list = ['MLP', ]
    data_to_write = [['repeat', 'model', 'fraction', 'max_iter', 'top_k', 'train_hit', 'test_hit']]
    for repeat in range(1):
        for use_data in 0, 1, 2:
            for model in model_list:
                for fraction in faction_list:
                    train_dataset, valid_dataset, test_dataset, diagnosis_index_dict, symptom_index_dict = \
                        read_data(structured_data_folder, minimum_symptom=1, read_from_cache=True, mode='english')
                    logger.info('train size: {}, test size: {}, valid size: {}'.format(
                        len(train_dataset), len(test_dataset), len(valid_dataset)))
                    symptom_num = len(train_dataset.symptom[0]) // 3
                    disease_num = len(diagnosis_index_dict)
                    assert symptom_num == 717
                    train_diagnosis, train_input, train_key = \
                        data_transform(train_dataset, fraction, use_data)
                    valid_diagnosis, valid_input, valid_key = \
                        data_transform(valid_dataset, fraction, use_data)
                    train_input = np.concatenate([train_input, valid_input], axis=0)
                    train_diagnosis = np.concatenate([train_diagnosis, valid_diagnosis], axis=0)
                    test_diagnosis, test_input, test_key = data_transform(test_dataset, 1.0, use_data)
                    clf = MLPClassifier(
                        hidden_layer_sizes=hidden_layer_sizes,
                        # random_state=random_state,
                        max_iter=max_iter,
                        learning_rate_init=0.0001,
                        batch_size=len(train_input) // 50,
                        learning_rate='adaptive'
                    )
                    logger.info('start training')
                    logger.info('use data type: {}, repeat: {}, model: {}'.format(use_data, repeat, model))
                    for i in range(max_iter):
                        clf.partial_fit(train_input, train_diagnosis, classes=np.unique(train_diagnosis))
                        if (i+1) % 50 == 0 or i == max_iter - 1:
                            train_diagnosis_prob = clf.predict_proba(train_input)
                            test_diagnosis_prob = clf.predict_proba(test_input)
                            for top_k in [1, 3, 5, 10, 20, 30, 50]:
                                test_top_k_hit = top_calculate(test_diagnosis, test_diagnosis_prob, top_k, disease_num)
                                train_top_k_hit = top_calculate(train_diagnosis, train_diagnosis_prob, top_k,
                                                                disease_num)
                                data_to_write.append([repeat, model,
                                                      fraction, max_iter, top_k, train_top_k_hit, test_top_k_hit])
                                logger.info('repeat: {:2d}, model: {:8s}, faction: {:3f}, iter {:6d}, use data: {},'
                                            ' train top {:2d} hit: {:5f}, test top {:2d} hit: {:5f}'
                                            .format(repeat, model, fraction, i+1, use_data,
                                                    top_k, train_top_k_hit, top_k,
                                                    test_top_k_hit))
                            logger.info('\n')
                    logger.info('training complete')
                    with open(data_path, 'w', encoding='utf-8-sig', newline='') as f:
                        csv.writer(f).writerows(data_to_write)


if __name__ == "__main__":
    main()


