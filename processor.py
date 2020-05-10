import os
import numpy as np
from time import perf_counter
from collections import Counter
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
import joblib
import matplotlib.pyplot as plt


def data_load(filepath):
    """ 加载数据 """
    start_time = perf_counter()
    movie_reviews = load_files(container_path=filepath, shuffle=True, encoding='UTF-8')
    print(f"原始数据加载完毕,耗时: {perf_counter() - start_time:0.4f}s")
    return movie_reviews


def data_show(data):
    """ 分析原始数据 """
    save_path = '/data1/usertest/Kolen/Homework/raw_data_distribution.png'
    print("数据大小:\t", len(data.data))
    print("标签名称:\t", data.target_names)
    print("标签编码:\t", np.unique(data.target))

    def add_labels(rects):
        """ 给数据标值 """
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2, height, height, ha='center', va='bottom')

    index = range(len(data.target_names))
    values = Counter(data.target).values()
    add_labels(plt.bar(index, values, width=0.8, color=['#0072BC', '#ED1C24'], tick_label=data.target_names))
    plt.title(u'Data Distribution')
    plt.ylim(ymin=0, ymax=1200)
    plt.savefig(save_path)
    plt.show()


stopwords = []  # 停用词列表


def data_preprocess(data):
    """ 数据预处理：去停用词、划分数据集"""
    stopwords_path = "/data1/usertest/Kolen/Homework/stopwords.txt"
    with open(stopwords_path) as f:
        for line in f.readlines():
            stopwords.append(line.strip('\n'))

    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=None)
    print(f"数据切分:\t [训练集] {len(x_train)}  [测试集] {len(x_test)}")
    return x_train, x_test, y_train, y_test


# 建立 vectorizer/classiffier Pipeline
models = {
    'SVM': Pipeline([
        ('vect', TfidfVectorizer(min_df=3, max_df=0.95)),  # stop_words=stopwords
        ('clf',  LinearSVC(max_iter=10000))
    ]),
    'Naive Bayes': Pipeline([
        ('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
        ('clf',  MultinomialNB())
    ]),
    'Logistic Regression': Pipeline([
        ('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
        ('clf',  LogisticRegression(max_iter=10000))
    ]),
    'KNN': Pipeline([
        ('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
        ('clf',  KNeighborsClassifier())
    ]),
    'Decision Tree': Pipeline([
        ('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
        ('clf',  DecisionTreeClassifier())
    ]),
    'Random Forest': Pipeline([
        ('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
        ('clf',  RandomForestClassifier(n_estimators=50))
    ]),
    'xgboost': Pipeline([
        ('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
        ('clf',  XGBClassifier(predictor='cpu_predictor'))
    ]),
    'AdaBoost': Pipeline([
        ('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
        ('clf', AdaBoostClassifier())
    ]),
    'MLP': Pipeline([
        ('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
        ('clf', MLPClassifier(early_stopping=True, max_iter=1000))
    ])
}

# 超参列表("__"前是模块名称，后是超参名称)  TODO: 调参
model_parameters = [
    {  # SVM
        'vect__ngram_range': [(1, 2), (1, 3), (2, 2)],
        'clf__C': np.logspace(-3, 3, 20),
    },
    {  # Naive Bayes
        'vect__ngram_range': [(1, 2), (1, 3), (2, 2)],
        'clf__alpha': np.logspace(-3, 3, 20),
    },
    {  # Logistic Regression
        'vect__ngram_range': [(1, 2), (1, 3), (2, 2)],
        'clf__C': np.logspace(-3, 3, 10),
        'clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    },
    {  # KNN
        'vect__ngram_range': [(1, 2), (1, 3), (2, 2)],
        'clf__n_neighbors': [2, 3, 5],
        'clf__leaf_size': [np.linspace(10, 30, 10)],
        'clf__weights': ['uniform', 'distance']
    },
    {  # Decision Tree
        'vect__ngram_range': [(1, 2), (1, 3), (2, 2)],
        'clf__max_depth': [2, 5, 8, 10, 15, 18, 20],
    },
    {  # Random Forest
        'vect__ngram_range': [(1, 2), (1, 3), (2, 2)],
        'clf__n_estimators': [30, 50, 60, 80, 100],
        'clf__max_depth': range(3, 14, 2),
        # 'clf__min_samples_split': range(50, 200, 10),
        # 'clf__min_samples_leaf': range(10, 60, 10),
        'clf__max_features': range(3, 11, 2)
    },
    # {  # xgboost
    #     # 'vect__ngram_range': [(1, 2), (1, 3), (2, 2)],
    #     # 'clf__max_depth': range(3, 25, 3),
    #     # 'clf__learning_rate': [0.001, 0.01, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4],
    #     # 'clf__gamma': [6, 9, 12, 13, 15],
    # },
    {  # AdaBoost
        'vect__ngram_range': [(1, 2), (1, 3), (2, 2)],
        'clf__n_estimators': [30, 50, 60, 80, 100],
        'clf__learning_rate': np.logspace(-3, 3, 10)
    },
    {  # MLP
        'vect__ngram_range': [(1, 2), (1, 3), (2, 2)],
        'clf__alpha': np.logspace(-3, 3, 10),
        'clf__learning_rate_init': np.logspace(-3, 3, 5),
        'clf__hidden_layer_sizes': [(256, ), (100, )],
    }
]


def depict_learning_curve(estimator, x_train, y_train, file_dir, model_name):
    """ 绘制学习曲线 """
    train_sizes, train_loss, test_loss = learning_curve(estimator, x_train, y_train, cv=5,
                                                        scoring='neg_mean_squared_error',
                                                        train_sizes=[0.1, 0.25, 0.5, 0.75, 1])
    train_loss_mean = -np.mean(train_loss, axis=1)
    test_loss_mean = -np.mean(test_loss, axis=1)

    plt.plot(train_sizes, train_loss_mean, 'o-', color="r", label="Training Loss")
    plt.plot(train_sizes, test_loss_mean, 'o-', color="g", label="Cross-validation Loss")

    plt.xlabel("Training examples")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.savefig(f'{file_dir}/Learning Curve of {model_name}.png')
    plt.show()


def run(data, x_train, x_test, y_train, y_test):
    """ 模型参数寻优(Grid Search)并训练预测 """
    for model_name, model, model_parameter in zip(models.keys(), models.values(), model_parameters):
        # 训练模型
        start_time = perf_counter()
        print('\n{:*^54}'.format(' ' + model_name + ' '))
        grid_search = GridSearchCV(estimator=model, param_grid=model_parameter, n_jobs=-1, cv=5, refit=True)
        grid_search.fit(x_train, y_train)
        print(f'- {model_name}模型拟合完毕, 耗时: {perf_counter() - start_time: 0.4f}s')  # 分析不同情况模型运行效率
        print(f'- 最佳参数组合: {grid_search.best_params_}')

        # 模型预测及效果评估
        y_predicted = grid_search.predict(x_test)
        print(f'- 评估报告:\n'
              f'{metrics.classification_report(y_test, y_predicted, target_names=data.target_names)}')

        # 困惑矩阵
        confusion_matrix = metrics.confusion_matrix(y_test, y_predicted)
        print(f'- 混淆矩阵:\n {confusion_matrix}')

        # 保存模型
        model_save_path = joblib.dump(grid_search, f"/data1/usertest/Kolen/Homework/model/feature/{model_name}.model")
        print(f'- 模型保存路径： {model_name}模型已保存在{model_save_path}')


if __name__ == "__main__":
    data_path = "/data1/usertest/Kolen/Homework/data/review_polarity/txt_sentoken"
    raw_data = data_load(data_path)  # 加载数据
    data_show(raw_data)  # 分析原始数据
    doc_terms_train, doc_terms_test, doc_label_train, doc_label_test = data_preprocess(raw_data)  # 数据清洗及划分
    run(raw_data, doc_terms_train, doc_terms_test, doc_label_train, doc_label_test)  # 模型训练预测

    # 绘制学习曲线
    model_dir = '/data1/usertest/Kolen/Homework/model/optimized'
    for _, _, file_names in os.walk(model_dir):
        for file_name in file_names:
            classifier = joblib.load(model_dir + '/' + file_name)
            depict_learning_curve(classifier, doc_terms_train, doc_label_train, model_dir, file_name)
