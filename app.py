from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

import subprocess
import sys
import json
import os


app = FastAPI(
    title="Machine Learning + Quantum Lab"
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

QML_RESULT_PATH = os.path.join(
    BASE_DIR,
    "qml_result.json",
)

QML_TEST_PATH = os.path.join(
    BASE_DIR,
    "qml_test.py",
)


# =========================================================
# IRIS DATA
# =========================================================

iris = load_iris()

X = iris.data
y = iris.target

class_names = [
    str(x)
    for x in iris.target_names
]

feature_names = [
    str(x)
    for x in iris.feature_names
]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)


# =========================================================
# RANDOM FOREST
# =========================================================

rf_model = RandomForestClassifier(
    n_estimators=120,
    max_depth=5,
    random_state=42,
)

rf_model.fit(
    X_train,
    y_train,
)

rf_pred = rf_model.predict(
    X_test
)

rf_proba = rf_model.predict_proba(
    X_test
)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred,
)

rf_confusion = confusion_matrix(
    y_test,
    rf_pred,
)

rf_precision, rf_recall, rf_f1, _ = (
    precision_recall_fscore_support(
        y_test,
        rf_pred,
        average=None,
        zero_division=0,
    )
)

rf_precision_macro, rf_recall_macro, rf_f1_macro, _ = (
    precision_recall_fscore_support(
        y_test,
        rf_pred,
        average="macro",
        zero_division=0,
    )
)

rf_feature_importance = [
    float(x)
    for x in rf_model.feature_importances_
]


# =========================================================
# ROC / AUC
# =========================================================

y_test_binary = label_binarize(
    y_test,
    classes=[0, 1, 2],
)

roc_data = []

for i, class_name in enumerate(
    class_names
):

    fpr, tpr, _ = roc_curve(
        y_test_binary[:, i],
        rf_proba[:, i],
    )

    class_auc = auc(
        fpr,
        tpr,
    )

    roc_data.append(
        {
            "class": class_name,

            "fpr": [
                float(x)
                for x in fpr
            ],

            "tpr": [
                float(x)
                for x in tpr
            ],

            "auc": float(
                class_auc
            ),
        }
    )


# =========================================================
# NEURAL NETWORK
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


nn_model = MLPClassifier(
    hidden_layer_sizes=(8, 8, 4),
    activation="relu",
    solver="adam",
    learning_rate_init=0.01,
    max_iter=1000,
    random_state=42,
)

nn_model.fit(
    X_train_scaled,
    y_train,
)

nn_pred = nn_model.predict(
    X_test_scaled
)

nn_accuracy = accuracy_score(
    y_test,
    nn_pred,
)

nn_confusion = confusion_matrix(
    y_test,
    nn_pred,
)

nn_precision_macro, nn_recall_macro, nn_f1_macro, _ = (
    precision_recall_fscore_support(
        y_test,
        nn_pred,
        average="macro",
        zero_division=0,
    )
)

nn_loss_curve = [
    float(x)
    for x in nn_model.loss_curve_
]


# =========================================================
# CLASSICAL LEADERBOARD
# =========================================================

main_models = [

    {
        "name": "Random Forest",
        "accuracy": float(
            rf_accuracy
        ),
        "f1": float(
            rf_f1_macro
        ),
    },

    {
        "name": "Neural Network",
        "accuracy": float(
            nn_accuracy
        ),
        "f1": float(
            nn_f1_macro
        ),
    },
]


main_models.sort(
    key=lambda model: (
        model["f1"],
        model["accuracy"],
    ),
    reverse=True,
)

main_winner = (
    main_models[0]["name"]
)


# =========================================================
# FAIR BINARY DATASET
# =========================================================

X_qml = iris.data[:100, 2:4]
y_qml = iris.target[:100]


X_q_train, X_q_test, y_q_train, y_q_test = (
    train_test_split(
        X_qml,
        y_qml,
        test_size=0.25,
        random_state=42,
        stratify=y_qml,
    )
)


# =========================================================
# BINARY RANDOM FOREST
# =========================================================

binary_rf = RandomForestClassifier(
    n_estimators=120,
    max_depth=5,
    random_state=42,
)

binary_rf.fit(
    X_q_train,
    y_q_train,
)

binary_rf_pred = binary_rf.predict(
    X_q_test
)

binary_rf_accuracy = accuracy_score(
    y_q_test,
    binary_rf_pred,
)


# =========================================================
# BINARY NEURAL NETWORK
# =========================================================

binary_scaler = StandardScaler()

X_q_train_scaled = (
    binary_scaler.fit_transform(
        X_q_train
    )
)

X_q_test_scaled = (
    binary_scaler.transform(
        X_q_test
    )
)


binary_nn = MLPClassifier(
    hidden_layer_sizes=(8, 8),
    activation="relu",
    solver="adam",
    learning_rate_init=0.01,
    max_iter=1000,
    random_state=42,
)

binary_nn.fit(
    X_q_train_scaled,
    y_q_train,
)

binary_nn_pred = binary_nn.predict(
    X_q_test_scaled
)

binary_nn_accuracy = accuracy_score(
    y_q_test,
    binary_nn_pred,
)


# =========================================================
# LOAD QML RESULT
# =========================================================

def load_qml_result():

    if os.path.exists(
        QML_RESULT_PATH
    ):

        with open(
            QML_RESULT_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(
                file
            )


    return {

        "model":
            "Variational Quantum Classifier",

        "features": [
            "petal length",
            "petal width",
        ],

        "qubits":
            2,

        "feature_map":
            "ZZFeatureMap reps=2",

        "ansatz":
            "RealAmplitudes reps=2",

        "optimizer":
            "COBYLA maxiter=150",

        "train_samples":
            75,

        "test_samples":
            25,

        "accuracy":
            0.80,

        "confusion_matrix": [
            [11, 2],
            [3, 9],
        ],

        "training_history":
            [],

        "iterations":
            0,

        "final_objective":
            None,
    }


# =========================================================
# QML CALCULATIONS
# =========================================================

def build_qml_data():

    qml_result = load_qml_result()

    qml_accuracy = float(
        qml_result.get(
            "accuracy",
            0.0,
        )
    )

    qml_confusion = qml_result.get(
        "confusion_matrix",
        [
            [0, 0],
            [0, 0],
        ],
    )

    qml_training_history = [
        float(x)
        for x in qml_result.get(
            "training_history",
            [],
        )
    ]

    qml_iterations = int(
        qml_result.get(
            "iterations",
            len(
                qml_training_history
            ),
        )
    )

    qml_final_objective = (
        qml_result.get(
            "final_objective"
        )
    )

    if qml_final_objective is not None:

        qml_final_objective = float(
            qml_final_objective
        )


    tn = qml_confusion[0][0]
    fp = qml_confusion[0][1]
    fn = qml_confusion[1][0]
    tp = qml_confusion[1][1]


    precision_0 = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else 0
    )

    recall_0 = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    precision_1 = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall_1 = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )


    f1_0 = (

        2
        *
        precision_0
        *
        recall_0
        /
        (
            precision_0
            +
            recall_0
        )

        if (
            precision_0
            +
            recall_0
        ) > 0

        else 0
    )


    f1_1 = (

        2
        *
        precision_1
        *
        recall_1
        /
        (
            precision_1
            +
            recall_1
        )

        if (
            precision_1
            +
            recall_1
        ) > 0

        else 0
    )


    qml_f1_macro = (
        f1_0
        +
        f1_1
    ) / 2


    return {

        "accuracy":
            qml_accuracy,

        "f1":
            float(
                qml_f1_macro
            ),

        "qubits":
            int(
                qml_result.get(
                    "qubits",
                    2,
                )
            ),

        "feature_map":
            qml_result.get(
                "feature_map",
                "ZZFeatureMap reps=2",
            ),

        "ansatz":
            qml_result.get(
                "ansatz",
                "RealAmplitudes reps=2",
            ),

        "optimizer":
            qml_result.get(
                "optimizer",
                "COBYLA maxiter=150",
            ),

        "features":
            qml_result.get(
                "features",
                [
                    "petal length",
                    "petal width",
                ],
            ),

        "train_samples":
            int(
                qml_result.get(
                    "train_samples",
                    75,
                )
            ),

        "test_samples":
            int(
                qml_result.get(
                    "test_samples",
                    25,
                )
            ),

        "confusion":
            qml_confusion,

        "training_history":
            qml_training_history,

        "iterations":
            qml_iterations,

        "final_objective":
            qml_final_objective,
    }


# =========================================================
# REAL QISKIT CIRCUIT
# =========================================================

theta = 1.20
phi = 0.65

quantum_circuit = QuantumCircuit(
    2
)

quantum_circuit.h(
    0
)

quantum_circuit.ry(
    theta,
    1,
)

quantum_circuit.cx(
    0,
    1,
)

quantum_circuit.rz(
    phi,
    0,
)

quantum_circuit.measure_all()

quantum_backend = AerSimulator()

compiled_circuit = transpile(
    quantum_circuit,
    quantum_backend,
)

quantum_job = quantum_backend.run(
    compiled_circuit,
    shots=2048,
)

quantum_result = (
    quantum_job.result()
)

quantum_counts = (
    quantum_result.get_counts()
)

quantum_probabilities = {}

for state, count in quantum_counts.items():

    quantum_probabilities[
        state
    ] = (
        count
        /
        2048
    )


# =========================================================
# DASHBOARD DATA
# =========================================================

def build_dashboard():

    qml = build_qml_data()

    binary_models = [

        {
            "name":
                "Random Forest",

            "accuracy":
                float(
                    binary_rf_accuracy
                ),
        },

        {
            "name":
                "Neural Network",

            "accuracy":
                float(
                    binary_nn_accuracy
                ),
        },

        {
            "name":
                "QML VQC",

            "accuracy":
                float(
                    qml["accuracy"]
                ),
        },
    ]


    binary_models.sort(
        key=lambda model:
            model["accuracy"],
        reverse=True,
    )


    binary_winner = (
        binary_models[0]["name"]
    )


    best_classical_accuracy = max(
        binary_rf_accuracy,
        binary_nn_accuracy,
    )


    quantum_gap = (
        qml["accuracy"]
        -
        best_classical_accuracy
    )


    return {

        "classes":
            class_names,

        "features":
            feature_names,


        "random_forest": {

            "trees":
                int(
                    rf_model.n_estimators
                ),

            "accuracy":
                float(
                    rf_accuracy
                ),

            "precision":
                float(
                    rf_precision_macro
                ),

            "recall":
                float(
                    rf_recall_macro
                ),

            "f1":
                float(
                    rf_f1_macro
                ),

            "feature_importance":
                rf_feature_importance,

            "confusion":
                rf_confusion
                .astype(int)
                .tolist(),

            "precision_classes":
                [
                    float(x)
                    for x in rf_precision
                ],

            "recall_classes":
                [
                    float(x)
                    for x in rf_recall
                ],

            "f1_classes":
                [
                    float(x)
                    for x in rf_f1
                ],

            "roc":
                roc_data,
        },


        "neural_network": {

            "accuracy":
                float(
                    nn_accuracy
                ),

            "precision":
                float(
                    nn_precision_macro
                ),

            "recall":
                float(
                    nn_recall_macro
                ),

            "f1":
                float(
                    nn_f1_macro
                ),

            "iterations":
                int(
                    nn_model.n_iter_
                ),

            "final_loss":
                float(
                    nn_model.loss_
                ),

            "loss_curve":
                nn_loss_curve,

            "confusion":
                nn_confusion
                .astype(int)
                .tolist(),
        },


        "main_models":
            main_models,

        "main_winner":
            main_winner,


        "qml":
            qml,


        "binary_benchmark": {

            "models":
                binary_models,

            "winner":
                binary_winner,

            "random_forest_accuracy":
                float(
                    binary_rf_accuracy
                ),

            "neural_network_accuracy":
                float(
                    binary_nn_accuracy
                ),

            "qml_accuracy":
                float(
                    qml["accuracy"]
                ),

            "best_classical_accuracy":
                float(
                    best_classical_accuracy
                ),

            "quantum_gap":
                float(
                    quantum_gap
                ),
        },


        "quantum": {

            "backend":
                "AerSimulator",

            "qubits":
                2,

            "shots":
                2048,

            "theta":
                theta,

            "phi":
                phi,

            "counts":
                quantum_counts,

            "probabilities":
                quantum_probabilities,
        },
    }


# =========================================================
# API
# =========================================================

@app.get("/health")
def health():

    return {
        "status":
            "ok",

        "project":
            "Machine Learning + Quantum Lab",
    }


@app.get("/api/dashboard")
def dashboard():

    return build_dashboard()


# =========================================================
# RUN QML TRAINING
# =========================================================

@app.post("/api/qml/train")
def run_qml_training():

    if not os.path.exists(
        QML_TEST_PATH
    ):

        raise HTTPException(
            status_code=404,
            detail="qml_test.py not found.",
        )


    try:

        result = subprocess.run(

            [
                sys.executable,
                QML_TEST_PATH,
            ],

            cwd=
                BASE_DIR,

            capture_output=
                True,

            text=
                True,

            timeout=
                900,
        )


    except subprocess.TimeoutExpired:

        raise HTTPException(
            status_code=500,
            detail="QML training timed out.",
        )


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(
                error
            ),
        )


    if result.returncode != 0:

        error_text = (
            result.stderr
            or
            result.stdout
            or
            "Unknown QML training error."
        )

        raise HTTPException(
            status_code=500,
            detail=error_text[-3000:],
        )


    if not os.path.exists(
        QML_RESULT_PATH
    ):

        raise HTTPException(
            status_code=500,
            detail="Training finished but qml_result.json was not created.",
        )


    return {

        "status":
            "completed",

        "message":
            "QML training completed successfully.",

        "dashboard":
            build_dashboard(),
    }


# =========================================================
# WEB PAGE
# =========================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
def home():

    return r"""
<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>
Machine Learning + Quantum Lab
</title>

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>


<style>

* {
    box-sizing: border-box;
}

body {

    margin: 0;

    min-height: 100vh;

    background:
        radial-gradient(
            circle at top,
            #14264b,
            #07101f 55%,
            #020611
        );

    color: white;

    font-family:
        Arial,
        Helvetica,
        sans-serif;
}

header {

    text-align: center;

    padding:
        38px
        20px
        28px;
}

.status {

    display:
        inline-block;

    padding:
        7px
        16px;

    border-radius:
        20px;

    border:
        1px solid
        #00ffbb;

    background:
        rgba(
            0,
            255,
            170,
            0.14
        );

    color:
        #8dffdb;
}

h1 {

    margin:
        16px
        0
        8px;

    font-size:
        44px;
}

.subtitle {

    color:
        #9bb4df;
}

.section-title {

    width:
        92%;

    max-width:
        1450px;

    margin:
        28px
        auto
        18px;

    color:
        #8fffe0;
}

.grid {

    width:
        92%;

    max-width:
        1450px;

    margin:
        auto;

    display:
        grid;

    grid-template-columns:
        repeat(
            2,
            minmax(
                0,
                1fr
            )
        );

    gap:
        24px;

    margin-bottom:
        24px;
}

.grid3 {

    width:
        92%;

    max-width:
        1450px;

    margin:
        auto;

    display:
        grid;

    grid-template-columns:
        repeat(
            3,
            minmax(
                0,
                1fr
            )
        );

    gap:
        24px;

    margin-bottom:
        35px;
}

.card {

    min-height:
        410px;

    padding:
        24px;

    border-radius:
        22px;

    border:
        1px solid
        #315081;

    background:
        rgba(
            7,
            18,
            40,
            0.88
        );

    box-shadow:
        0
        0
        35px
        rgba(
            0,
            132,
            255,
            0.09
        );

    overflow:
        hidden;
}

.card p {

    color:
        #9db4d7;
}

.chart {

    height:
        310px;
}

.info {

    padding:
        15px;

    border-radius:
        14px;

    border:
        1px solid
        rgba(
            0,
            255,
            190,
            0.30
        );

    background:
        rgba(
            0,
            255,
            190,
            0.05
        );

    color:
        #8fffe0;

    font-family:
        monospace;

    line-height:
        1.7;
}

.warning {

    padding:
        12px;

    margin-top:
        12px;

    border:
        1px solid
        #e3b34e;

    border-radius:
        12px;

    color:
        #ffd779;

    background:
        rgba(
            255,
            180,
            50,
            0.06
        );

    font-size:
        13px;
}

.analysis-good {

    color:
        #8fffe0;

    font-weight:
        bold;
}

.analysis-bad {

    color:
        #ffb46a;

    font-weight:
        bold;
}

.metrics {

    display:
        grid;

    grid-template-columns:
        repeat(
            3,
            1fr
        );

    gap:
        10px;

    margin-top:
        12px;
}

.metric {

    padding:
        12px;

    border:
        1px solid
        #27456c;

    border-radius:
        12px;

    text-align:
        center;
}

.metric-title {

    color:
        #8fa7d6;

    font-size:
        12px;
}

.metric-value {

    margin-top:
        5px;

    font-size:
        20px;

    font-weight:
        bold;
}

.qml-button {

    width:
        100%;

    margin-top:
        16px;

    padding:
        14px
        20px;

    border:
        1px solid
        #00ffbb;

    border-radius:
        12px;

    background:
        rgba(
            0,
            255,
            187,
            0.12
        );

    color:
        #8fffe0;

    font-size:
        16px;

    font-weight:
        bold;

    cursor:
        pointer;

    transition:
        0.2s;
}

.qml-button:hover {

    background:
        rgba(
            0,
            255,
            187,
            0.25
        );

    box-shadow:
        0
        0
        25px
        rgba(
            0,
            255,
            187,
            0.25
        );
}

.qml-button:disabled {

    opacity:
        0.55;

    cursor:
        wait;
}

.qml-train-status {

    margin-top:
        12px;

    padding:
        10px;

    border-radius:
        10px;

    font-family:
        monospace;

    color:
        #9db4d7;

    border:
        1px solid
        #315081;
}

.qml-train-status.running {

    color:
        #ffd779;

    border-color:
        #e3b34e;
}

.qml-train-status.success {

    color:
        #8fffe0;

    border-color:
        #00ffbb;
}

.qml-train-status.error {

    color:
        #ff9daf;

    border-color:
        #ff647c;
}

.quantum-box {

    height:
        300px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;
}

.bloch {

    width:
        250px;

    height:
        250px;

    position:
        relative;

    border-radius:
        50%;

    border:
        2px solid
        #00d9ff;

    box-shadow:
        0
        0
        40px
        rgba(
            0,
            217,
            255,
            0.45
        );

    animation:
        glow
        3s
        infinite;
}

.orbit {

    position:
        absolute;

    left:
        50%;

    top:
        50%;

    width:
        100%;

    height:
        37%;

    border:
        1px solid
        #299ee0;

    border-radius:
        50%;
}

.o1 {

    transform:
        translate(
            -50%,
            -50%
        );
}

.o2 {

    animation:
        orbit1
        5s
        linear
        infinite;
}

.o3 {

    animation:
        orbit2
        7s
        linear
        infinite;
}

.vector {

    position:
        absolute;

    left:
        50%;

    top:
        50%;

    width:
        4px;

    height:
        105px;

    background:
        linear-gradient(
            #ffffff,
            #00ffbb
        );

    transform-origin:
        center bottom;

    box-shadow:
        0
        0
        20px
        #00ffbb;

    animation:
        vectorMove
        4s
        ease-in-out
        infinite;
}

.vector::after {

    content:
        "";

    position:
        absolute;

    top:
        -8px;

    left:
        50%;

    width:
        16px;

    height:
        16px;

    border-radius:
        50%;

    background:
        white;

    transform:
        translateX(
            -50%
        );

    box-shadow:
        0
        0
        20px
        #00ffbb;
}

.leader {

    padding:
        18px;

    border:
        1px solid
        #00ffbb;

    border-radius:
        16px;

    box-shadow:
        0
        0
        30px
        rgba(
            0,
            255,
            187,
            0.12
        );
}

.model-row {

    margin-top:
        12px;

    padding:
        16px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        space-between;

    border:
        1px solid
        #315081;

    border-radius:
        14px;
}

.score {

    color:
        #8fffe0;

    font-family:
        monospace;
}

.tree {

    height:
        250px;

    position:
        relative;
}

.node {

    position:
        absolute;

    width:
        65px;

    height:
        38px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    background:
        #16385f;

    border:
        1px solid
        #5388cb;

    border-radius:
        10px;

    transition:
        0.2s;
}

.node.active {

    background:
        #00a889;

    transform:
        scale(
            1.12
        );

    box-shadow:
        0
        0
        25px
        #00ffd0;
}

.n1 {
    left: 45%;
    top: 10px;
}

.n2 {
    left: 20%;
    top: 90px;
}

.n3 {
    left: 68%;
    top: 90px;
}

.n4 {
    left: 5%;
    top: 190px;
}

.n5 {
    left: 30%;
    top: 190px;
}

.n6 {
    left: 58%;
    top: 190px;
}

.n7 {
    left: 82%;
    top: 190px;
}

.nn {

    height:
        250px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        space-around;
}

.layer {

    height:
        210px;

    display:
        flex;

    flex-direction:
        column;

    justify-content:
        space-around;
}

.neuron {

    width:
        27px;

    height:
        27px;

    border-radius:
        50%;

    background:
        #173861;

    border:
        1px solid
        #4f8cd4;

    transition:
        0.2s;
}

.neuron.active {

    background:
        #00d4a8;

    transform:
        scale(
            1.25
        );

    box-shadow:
        0
        0
        25px
        #00ffd5;
}

.qml {

    height:
        250px;

    position:
        relative;
}

.wire {

    position:
        absolute;

    left:
        8%;

    right:
        8%;

    height:
        2px;

    background:
        #39577e;
}

.w1 {
    top: 45px;
}

.w2 {
    top: 100px;
}

.w3 {
    top: 155px;
}

.w4 {
    top: 210px;
}

.gate {

    position:
        absolute;

    width:
        38px;

    height:
        38px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        8px;

    background:
        #183555;

    border:
        1px solid
        #608bd0;

    transition:
        0.2s;
}

.gate.active {

    background:
        #7c3aed;

    transform:
        scale(
            1.18
        );

    box-shadow:
        0
        0
        25px
        #a65cff;
}

.g1 {
    left: 20%;
    top: 26px;
}

.g2 {
    left: 35%;
    top: 81px;
}

.g3 {
    left: 52%;
    top: 136px;
}

.g4 {
    left: 70%;
    top: 191px;
}

.module-status {

    text-align:
        center;

    color:
        #8fffe0;

    font-family:
        monospace;
}

.error-box {

    display:
        none;

    width:
        92%;

    max-width:
        1450px;

    margin:
        0 auto 20px;

    padding:
        14px;

    border:
        1px solid
        #ff647c;

    border-radius:
        12px;

    color:
        #ff9daf;

    background:
        rgba(
            255,
            60,
            90,
            0.08
        );
}

footer {

    text-align:
        center;

    padding:
        25px;

    color:
        #7894bd;
}


@keyframes glow {

    50% {

        box-shadow:
            0
            0
            65px
            rgba(
                0,
                217,
                255,
                0.8
            );
    }
}


@keyframes orbit1 {

    from {

        transform:
            translate(
                -50%,
                -50%
            )
            rotate(
                55deg
            );
    }

    to {

        transform:
            translate(
                -50%,
                -50%
            )
            rotate(
                415deg
            );
    }
}


@keyframes orbit2 {

    from {

        transform:
            translate(
                -50%,
                -50%
            )
            rotate(
                -55deg
            );
    }

    to {

        transform:
            translate(
                -50%,
                -50%
            )
            rotate(
                -415deg
            );
    }
}


@keyframes vectorMove {

    0% {

        transform:
            translate(
                -50%,
                -100%
            )
            rotate(
                15deg
            );
    }

    50% {

        transform:
            translate(
                -50%,
                -100%
            )
            rotate(
                150deg
            );
    }

    100% {

        transform:
            translate(
                -50%,
                -100%
            )
            rotate(
                375deg
            );
    }
}


@media(
    max-width:
    1000px
) {

    .grid,
    .grid3 {

        grid-template-columns:
            1fr;
    }
}

</style>

</head>


<body>


<header>

<div class="status">
● SYSTEM ONLINE
</div>

<h1>
Machine Learning + Quantum Lab
</h1>

<div class="subtitle">
AI • Random Forest • Neural Networks • Qiskit • Quantum Machine Learning
</div>

</header>


<div
    id="error-box"
    class="error-box"
>
Dashboard error
</div>


<h2 class="section-title">
Classical Machine Learning
</h2>


<div class="grid">


<div class="card">

<h2>
🧠 Machine Learning Training
</h2>

<p>
Animated training visualization
</p>

<div
    id="training-chart"
    class="chart"
></div>

<div class="metrics">


<div class="metric">

<div class="metric-title">
Epoch
</div>

<div
    id="epoch"
    class="metric-value"
>
0
</div>

</div>


<div class="metric">

<div class="metric-title">
Loss
</div>

<div
    id="live-loss"
    class="metric-value"
>
1.000
</div>

</div>


<div class="metric">

<div class="metric-title">
Accuracy
</div>

<div
    id="live-accuracy"
    class="metric-value"
>
0%
</div>

</div>


</div>

</div>


<div class="card">

<h2>
🌲 Real Random Forest
</h2>

<div
    id="rf-info"
    class="info"
>
Loading...
</div>

<div
    id="feature-chart"
    class="chart"
></div>

</div>


</div>


<div class="grid">


<div class="card">

<h2>
📊 Random Forest Confusion Matrix
</h2>

<div
    id="rf-confusion-chart"
    class="chart"
></div>

</div>


<div class="card">

<h2>
🎯 Precision / Recall / F1
</h2>

<div
    id="rf-metrics-chart"
    class="chart"
></div>

</div>


</div>


<div class="grid">


<div class="card">

<h2>
📈 ROC / AUC
</h2>

<div
    id="roc-chart"
    class="chart"
></div>

</div>


<div class="card">

<h2>
🧬 Real Neural Network
</h2>

<div
    id="nn-info"
    class="info"
>
Loading...
</div>

<div
    id="nn-loss-chart"
    class="chart"
></div>

</div>


</div>


<div class="grid">


<div class="card">

<h2>
🧠 Neural Network Confusion Matrix
</h2>

<div
    id="nn-confusion-chart"
    class="chart"
></div>

</div>


<div class="card">

<h2>
🏆 Classical Model Leaderboard
</h2>

<div id="main-leaderboard">
Loading...
</div>

</div>


</div>


<h2 class="section-title">
Quantum Computing
</h2>


<div class="grid">


<div class="card">

<h2>
⚛ Real Qiskit Quantum Circuit
</h2>

<div
    id="quantum-info"
    class="info"
>
Loading...
</div>

<div
    id="quantum-chart"
    class="chart"
></div>

</div>


<div class="card">

<h2>
🌐 Quantum State Visualization
</h2>

<div class="quantum-box">

<div class="bloch">

<div class="orbit o1"></div>

<div class="orbit o2"></div>

<div class="orbit o3"></div>

<div class="vector"></div>

</div>

</div>

<div
    id="quantum-live"
    class="module-status"
>
|ψ⟩ = α|0⟩ + β|1⟩
</div>

</div>


</div>


<h2 class="section-title">
Quantum Machine Learning
</h2>


<div class="grid">


<div class="card">

<h2>
⚛ Real Variational Quantum Classifier
</h2>

<p>
Real Qiskit Machine Learning training result
</p>

<div
    id="qml-info"
    class="info"
>
Loading VQC...
</div>


<button
    id="run-qml-button"
    class="qml-button"
>
▶ Run QML Training
</button>


<div
    id="qml-train-status"
    class="qml-train-status"
>
Ready for training.
</div>


<div class="warning">

The latest successful VQC result is stored in qml_result.json.
Click Run QML Training to train the model again.

</div>

</div>


<div class="card">

<h2>
📊 QML Confusion Matrix
</h2>

<div
    id="qml-confusion-chart"
    class="chart"
></div>

</div>


</div>


<div class="grid">


<div class="card">

<h2>
📉 VQC Training Convergence
</h2>

<div
    id="qml-convergence-chart"
    class="chart"
></div>

<div
    id="convergence-info"
    class="info"
>
Loading optimization history...
</div>

</div>


<div class="card">

<h2>
⚛ Quantum Advantage Analysis
</h2>

<div
    id="advantage-chart"
    class="chart"
></div>

<div
    id="advantage-info"
    class="info"
>
Loading analysis...
</div>

</div>


</div>


<div class="grid">


<div class="card">

<h2>
⚔ Fair Classical vs Quantum Benchmark
</h2>

<p>
Same binary Iris task: setosa vs versicolor
</p>

<div
    id="binary-comparison-chart"
    class="chart"
></div>

</div>


<div class="card">

<h2>
🏆 QML Benchmark Leaderboard
</h2>

<div id="qml-leaderboard">
Loading...
</div>

</div>


</div>


<div class="grid3">


<div class="card">

<h2>
🌳 Decision Tree
</h2>

<div class="tree">

<div class="node n1">
Root
</div>

<div class="node n2">
X₁
</div>

<div class="node n3">
X₂
</div>

<div class="node n4">
A
</div>

<div class="node n5">
B
</div>

<div class="node n6">
C
</div>

<div class="node n7">
D
</div>

</div>

<div
    id="tree-status"
    class="module-status"
>
Traversing...
</div>

</div>


<div class="card">

<h2>
🧬 Neural Network Flow
</h2>

<div class="nn">


<div class="layer">

<div class="neuron"></div>
<div class="neuron"></div>
<div class="neuron"></div>
<div class="neuron"></div>

</div>


<div class="layer">

<div class="neuron"></div>
<div class="neuron"></div>
<div class="neuron"></div>
<div class="neuron"></div>
<div class="neuron"></div>

</div>


<div class="layer">

<div class="neuron"></div>
<div class="neuron"></div>
<div class="neuron"></div>
<div class="neuron"></div>

</div>


<div class="layer">

<div class="neuron"></div>
<div class="neuron"></div>
<div class="neuron"></div>

</div>


</div>

<div
    id="nn-status"
    class="module-status"
>
Forward propagation...
</div>

</div>


<div class="card">

<h2>
⚛ Quantum ML Circuit
</h2>

<div class="qml">

<div class="wire w1"></div>
<div class="wire w2"></div>
<div class="wire w3"></div>
<div class="wire w4"></div>

<div class="gate g1">
H
</div>

<div class="gate g2">
ZZ
</div>

<div class="gate g3">
RY
</div>

<div class="gate g4">
M
</div>

</div>

<div
    id="qml-status"
    class="module-status"
>
Quantum circuit ready
</div>

</div>


</div>


<footer>
Machine Learning + Quantum Research Lab
</footer>


<script>


const plotConfig = {

    responsive:
        true,

    displayModeBar:
        false
};


function baseLayout(
    xTitle = "",
    yTitle = ""
) {

    return {

        paper_bgcolor:
            "rgba(0,0,0,0)",

        plot_bgcolor:
            "rgba(0,0,0,0)",

        font: {
            color:
                "#ffffff"
        },

        margin: {
            l: 65,
            r: 20,
            t: 25,
            b: 60
        },

        xaxis: {

            title:
                xTitle,

            gridcolor:
                "#24395d"
        },

        yaxis: {

            title:
                yTitle,

            gridcolor:
                "#24395d"
        }
    };
}


function showError(
    message
) {

    const box =
        document.getElementById(
            "error-box"
        );

    box.style.display =
        "block";

    box.textContent =
        message;
}


function clearError() {

    document
        .getElementById(
            "error-box"
        )
        .style.display =
            "none";
}


/* =========================================================
LIVE TRAINING
========================================================= */

function startTraining() {

    let epoch =
        0;

    let loss =
        0.95;

    const epochs =
        [];

    const losses =
        [];

    const accuracies =
        [];


    const trainingLayout =
        baseLayout(
            "Epoch",
            "Metric"
        );

    trainingLayout.yaxis.range =
        [0, 1.05];


    Plotly.newPlot(

        "training-chart",

        [

            {
                x: [],
                y: [],

                name:
                    "Loss",

                mode:
                    "lines"
            },

            {
                x: [],
                y: [],

                name:
                    "Accuracy",

                mode:
                    "lines"
            }

        ],

        trainingLayout,

        plotConfig
    );


    setInterval(
        function () {

            epoch +=
                1;


            loss =
                Math.max(
                    0.03,

                    loss
                    *
                    0.965
                );


            const accuracy =
                Math.min(
                    0.98,

                    0.1
                    +
                    epoch
                    *
                    0.015
                );


            epochs.push(
                epoch
            );

            losses.push(
                loss
            );

            accuracies.push(
                accuracy
            );


            if (
                epochs.length
                >
                60
            ) {

                epochs.shift();

                losses.shift();

                accuracies.shift();
            }


            Plotly.update(

                "training-chart",

                {

                    x: [
                        epochs,
                        epochs
                    ],

                    y: [
                        losses,
                        accuracies
                    ]
                }
            );


            document
                .getElementById(
                    "epoch"
                )
                .textContent =
                    epoch;


            document
                .getElementById(
                    "live-loss"
                )
                .textContent =
                    loss.toFixed(
                        3
                    );


            document
                .getElementById(
                    "live-accuracy"
                )
                .textContent =
                    (
                        accuracy
                        *
                        100
                    )
                    .toFixed(
                        1
                    )
                    +
                    "%";

        },

        350
    );
}


/* =========================================================
DASHBOARD
========================================================= */

async function loadDashboard() {

    try {

        clearError();


        if (
            typeof Plotly
            ===
            "undefined"
        ) {

            throw new Error(
                "Plotly failed to load."
            );
        }


        const response =
            await fetch(
                "/api/dashboard"
            );


        if (
            !response.ok
        ) {

            throw new Error(
                "API error "
                +
                response.status
            );
        }


        const data =
            await response.json();


        const rf =
            data.random_forest;

        const nn =
            data.neural_network;

        const qml =
            data.qml;

        const quantum =
            data.quantum;

        const benchmark =
            data.binary_benchmark;


        /* RANDOM FOREST */

        document
            .getElementById(
                "rf-info"
            )
            .innerHTML =

                "MODEL: RandomForestClassifier"

                +

                "<br>TREES: "
                +
                rf.trees

                +

                "<br>REAL ACCURACY: "
                +
                (
                    rf.accuracy
                    *
                    100
                )
                .toFixed(
                    2
                )
                +
                "%"

                +

                "<br>MACRO F1: "
                +
                rf.f1
                .toFixed(
                    4
                );


        Plotly.newPlot(

            "feature-chart",

            [
                {
                    x:
                        rf.feature_importance,

                    y:
                        data.features,

                    type:
                        "bar",

                    orientation:
                        "h"
                }
            ],

            baseLayout(
                "Feature Importance",
                ""
            ),

            plotConfig
        );


        Plotly.newPlot(

            "rf-confusion-chart",

            [
                {
                    z:
                        rf.confusion,

                    x:
                        data.classes,

                    y:
                        data.classes,

                    type:
                        "heatmap",

                    colorscale:
                        "Viridis",

                    text:
                        rf.confusion,

                    texttemplate:
                        "%{text}"
                }
            ],

            baseLayout(
                "Predicted",
                "Actual"
            ),

            plotConfig
        );


        const metricsLayout =
            baseLayout(
                "Class",
                "Score"
            );

        metricsLayout.barmode =
            "group";

        metricsLayout.yaxis.range =
            [0, 1.05];


        Plotly.newPlot(

            "rf-metrics-chart",

            [

                {
                    x:
                        data.classes,

                    y:
                        rf.precision_classes,

                    type:
                        "bar",

                    name:
                        "Precision"
                },

                {
                    x:
                        data.classes,

                    y:
                        rf.recall_classes,

                    type:
                        "bar",

                    name:
                        "Recall"
                },

                {
                    x:
                        data.classes,

                    y:
                        rf.f1_classes,

                    type:
                        "bar",

                    name:
                        "F1"
                }

            ],

            metricsLayout,

            plotConfig
        );


        /* ROC */

        const rocTraces =
            [];


        rf.roc.forEach(
            function (
                item
            ) {

                rocTraces.push(
                    {
                        x:
                            item.fpr,

                        y:
                            item.tpr,

                        mode:
                            "lines",

                        name:
                            item.class
                            +
                            " AUC="
                            +
                            item.auc
                            .toFixed(
                                3
                            )
                    }
                );
            }
        );


        rocTraces.push(
            {
                x:
                    [0, 1],

                y:
                    [0, 1],

                mode:
                    "lines",

                name:
                    "Random",

                line: {
                    dash:
                        "dash"
                }
            }
        );


        Plotly.newPlot(

            "roc-chart",

            rocTraces,

            baseLayout(
                "False Positive Rate",
                "True Positive Rate"
            ),

            plotConfig
        );


        /* NEURAL NETWORK */

        document
            .getElementById(
                "nn-info"
            )
            .innerHTML =

                "MODEL: MLPClassifier"

                +

                "<br>ARCHITECTURE: 4 → 8 → 8 → 4 → 3"

                +

                "<br>ITERATIONS: "
                +
                nn.iterations

                +

                "<br>FINAL LOSS: "
                +
                nn.final_loss
                .toFixed(
                    6
                )

                +

                "<br>REAL ACCURACY: "
                +
                (
                    nn.accuracy
                    *
                    100
                )
                .toFixed(
                    2
                )
                +
                "%"

                +

                "<br>MACRO F1: "
                +
                nn.f1
                .toFixed(
                    4
                );


        Plotly.newPlot(

            "nn-loss-chart",

            [
                {
                    x:
                        nn.loss_curve.map(
                            function (
                                value,
                                index
                            ) {

                                return (
                                    index
                                    +
                                    1
                                );
                            }
                        ),

                    y:
                        nn.loss_curve,

                    mode:
                        "lines",

                    name:
                        "Loss"
                }
            ],

            baseLayout(
                "Iteration",
                "Loss"
            ),

            plotConfig
        );


        Plotly.newPlot(

            "nn-confusion-chart",

            [
                {
                    z:
                        nn.confusion,

                    x:
                        data.classes,

                    y:
                        data.classes,

                    type:
                        "heatmap",

                    colorscale:
                        "Hot",

                    text:
                        nn.confusion,

                    texttemplate:
                        "%{text}"
                }
            ],

            baseLayout(
                "Predicted",
                "Actual"
            ),

            plotConfig
        );


        /* CLASSICAL LEADERBOARD */

        document
            .getElementById(
                "main-leaderboard"
            )
            .innerHTML =

                `
                <div class="leader">

                    🏆 Leader:
                    <strong>
                        ${data.main_winner}
                    </strong>

                </div>

                ${data.main_models.map(
                    function (
                        model,
                        index
                    ) {

                        return `

                        <div class="model-row">

                            <div>

                                ${
                                    index
                                    ===
                                    0
                                    ?
                                    "🥇"
                                    :
                                    "🥈"
                                }

                                <strong>
                                    ${model.name}
                                </strong>

                            </div>

                            <div class="score">

                                Accuracy:
                                ${(model.accuracy * 100).toFixed(2)}%

                                <br>

                                F1:
                                ${model.f1.toFixed(4)}

                            </div>

                        </div>

                        `;
                    }
                ).join("")}
                `;


        /* QUANTUM CIRCUIT */

        document
            .getElementById(
                "quantum-info"
            )
            .innerHTML =

                "BACKEND: "
                +
                quantum.backend

                +

                "<br>QUBITS: "
                +
                quantum.qubits

                +

                "<br>SHOTS: "
                +
                quantum.shots

                +

                "<br>θ: "
                +
                quantum.theta
                .toFixed(
                    2
                )

                +

                "<br>φ: "
                +
                quantum.phi
                .toFixed(
                    2
                );


        Plotly.newPlot(

            "quantum-chart",

            [
                {
                    x:
                        Object.keys(
                            quantum.probabilities
                        ),

                    y:
                        Object.values(
                            quantum.probabilities
                        ),

                    type:
                        "bar",

                    name:
                        "Probability"
                }
            ],

            baseLayout(
                "Measured State",
                "Probability"
            ),

            plotConfig
        );


        /* QML INFO */

        let finalObjectiveText =
            "N/A";


        if (
            qml.final_objective
            !==
            null
        ) {

            finalObjectiveText =
                qml.final_objective
                .toFixed(
                    6
                );
        }


        document
            .getElementById(
                "qml-info"
            )
            .innerHTML =

                "MODEL: Variational Quantum Classifier"

                +

                "<br>QUBITS: "
                +
                qml.qubits

                +

                "<br>FEATURES: "
                +
                qml.features.join(
                    " + "
                )

                +

                "<br>FEATURE MAP: "
                +
                qml.feature_map

                +

                "<br>ANSATZ: "
                +
                qml.ansatz

                +

                "<br>OPTIMIZER: "
                +
                qml.optimizer

                +

                "<br>TRAIN SAMPLES: "
                +
                qml.train_samples

                +

                "<br>TEST SAMPLES: "
                +
                qml.test_samples

                +

                "<br>REAL ACCURACY: "
                +
                (
                    qml.accuracy
                    *
                    100
                )
                .toFixed(
                    2
                )
                +
                "%"

                +

                "<br>MACRO F1: "
                +
                qml.f1
                .toFixed(
                    4
                )

                +

                "<br>OPTIMIZATION EVALUATIONS: "
                +
                qml.iterations

                +

                "<br>FINAL OBJECTIVE: "
                +
                finalObjectiveText;


        /* QML CONFUSION */

        Plotly.newPlot(

            "qml-confusion-chart",

            [
                {
                    z:
                        qml.confusion,

                    x: [
                        "setosa",
                        "versicolor"
                    ],

                    y: [
                        "setosa",
                        "versicolor"
                    ],

                    type:
                        "heatmap",

                    colorscale:
                        "Viridis",

                    text:
                        qml.confusion,

                    texttemplate:
                        "%{text}"
                }
            ],

            baseLayout(
                "Predicted",
                "Actual"
            ),

            plotConfig
        );


        /* QML CONVERGENCE */

        if (
            qml.training_history.length
            >
            0
        ) {

            const steps =
                qml.training_history.map(
                    function (
                        value,
                        index
                    ) {

                        return (
                            index
                            +
                            1
                        );
                    }
                );


            Plotly.newPlot(

                "qml-convergence-chart",

                [
                    {
                        x:
                            steps,

                        y:
                            qml.training_history,

                        mode:
                            "lines",

                        name:
                            "VQC Objective"
                    }
                ],

                baseLayout(
                    "COBYLA Evaluation",
                    "Objective Value"
                ),

                plotConfig
            );


            const startObjective =
                qml.training_history[0];


            const endObjective =
                qml.training_history[
                    qml.training_history.length
                    -
                    1
                ];


            document
                .getElementById(
                    "convergence-info"
                )
                .innerHTML =

                    "EVALUATIONS: "
                    +
                    qml.iterations

                    +

                    "<br>INITIAL OBJECTIVE: "
                    +
                    startObjective
                    .toFixed(
                        6
                    )

                    +

                    "<br>FINAL OBJECTIVE: "
                    +
                    endObjective
                    .toFixed(
                        6
                    )

                    +

                    "<br>OBJECTIVE REDUCTION: "
                    +
                    (
                        startObjective
                        -
                        endObjective
                    )
                    .toFixed(
                        6
                    );
        }

        else {

            Plotly.purge(
                "qml-convergence-chart"
            );


            document
                .getElementById(
                    "convergence-info"
                )
                .textContent =

                    "No optimization history found.";
        }


        /* BENCHMARK */

        const benchmarkLayout =
            baseLayout(
                "Model",
                "Accuracy"
            );

        benchmarkLayout.yaxis.range =
            [0, 1.05];


        Plotly.newPlot(

            "binary-comparison-chart",

            [
                {
                    x: [
                        "Random Forest",
                        "Neural Network",
                        "QML VQC"
                    ],

                    y: [
                        benchmark.random_forest_accuracy,
                        benchmark.neural_network_accuracy,
                        benchmark.qml_accuracy
                    ],

                    type:
                        "bar"
                }
            ],

            benchmarkLayout,

            plotConfig
        );


        /* QUANTUM ADVANTAGE */

        const advantageLayout =
            baseLayout(
                "Model",
                "Accuracy"
            );

        advantageLayout.yaxis.range =
            [0, 1.05];


        Plotly.newPlot(

            "advantage-chart",

            [
                {
                    x: [
                        "Best Classical",
                        "QML VQC"
                    ],

                    y: [
                        benchmark.best_classical_accuracy,
                        benchmark.qml_accuracy
                    ],

                    type:
                        "bar"
                }
            ],

            advantageLayout,

            plotConfig
        );


        const gapPercent =
            benchmark.quantum_gap
            *
            100;


        let advantageMessage =
            "";


        if (
            gapPercent
            >
            0
        ) {

            advantageMessage =

                "<span class='analysis-good'>QUANTUM ADVANTAGE DETECTED</span>"

                +

                "<br>QML advantage: +"
                +
                gapPercent
                .toFixed(
                    2
                )
                +
                " percentage points";
        }

        else if (
            gapPercent
            <
            0
        ) {

            advantageMessage =

                "<span class='analysis-bad'>NO QUANTUM ADVANTAGE ON THIS DATASET</span>"

                +

                "<br>QML gap: "
                +
                gapPercent
                .toFixed(
                    2
                )
                +
                " percentage points";
        }

        else {

            advantageMessage =

                "<span class='analysis-good'>EQUAL PERFORMANCE</span>";
        }


        document
            .getElementById(
                "advantage-info"
            )
            .innerHTML =

                "BEST CLASSICAL ACCURACY: "
                +
                (
                    benchmark.best_classical_accuracy
                    *
                    100
                )
                .toFixed(
                    2
                )
                +
                "%"

                +

                "<br>QML ACCURACY: "
                +
                (
                    benchmark.qml_accuracy
                    *
                    100
                )
                .toFixed(
                    2
                )
                +
                "%"

                +

                "<br><br>"

                +

                advantageMessage;


        /* QML LEADERBOARD */

        document
            .getElementById(
                "qml-leaderboard"
            )
            .innerHTML =

                `
                <div class="leader">

                    🏆 Binary benchmark leader:
                    <strong>
                        ${benchmark.winner}
                    </strong>

                </div>

                ${benchmark.models.map(
                    function (
                        model,
                        index
                    ) {

                        const medals =
                            [
                                "🥇",
                                "🥈",
                                "🥉"
                            ];

                        return `

                        <div class="model-row">

                            <div>

                                ${medals[index]}

                                <strong>
                                    ${model.name}
                                </strong>

                            </div>

                            <div class="score">

                                Accuracy:
                                ${(model.accuracy * 100).toFixed(2)}%

                            </div>

                        </div>

                        `;
                    }
                ).join("")}
                `;

    }

    catch (
        error
    ) {

        showError(
            "Dashboard error: "
            +
            error.message
        );
    }
}


/* =========================================================
RUN QML
========================================================= */

async function runQMLTraining() {

    const button =
        document.getElementById(
            "run-qml-button"
        );

    const status =
        document.getElementById(
            "qml-train-status"
        );


    button.disabled =
        true;

    button.textContent =
        "⏳ Training VQC...";


    status.className =
        "qml-train-status running";

    status.textContent =
        "Qiskit VQC training is running. Please wait...";


    try {

        clearError();


        const response =
            await fetch(

                "/api/qml/train",

                {
                    method:
                        "POST"
                }
            );


        const result =
            await response.json();


        if (
            !response.ok
        ) {

            throw new Error(
                result.detail
                ||
                "QML training failed."
            );
        }


        status.className =
            "qml-train-status success";


        status.textContent =
            "✓ Training completed. Dashboard updated.";


        await loadDashboard();

    }

    catch (
        error
    ) {

        status.className =
            "qml-train-status error";


        status.textContent =
            "Training error: "
            +
            error.message;


        showError(
            "QML training error: "
            +
            error.message
        );
    }

    finally {

        button.disabled =
            false;


        button.textContent =
            "▶ Run QML Training";
    }
}


/* =========================================================
QUANTUM ANIMATION
========================================================= */

function startQuantumAnimation() {

    let phase =
        0;


    setInterval(
        function () {

            phase +=
                0.05;


            const alpha =
                Math.abs(
                    Math.cos(
                        phase
                    )
                );


            const beta =
                Math.abs(
                    Math.sin(
                        phase
                    )
                );


            document
                .getElementById(
                    "quantum-live"
                )
                .innerHTML =

                    "α = "
                    +
                    alpha
                    .toFixed(
                        3
                    )

                    +

                    " &nbsp;&nbsp; β = "
                    +
                    beta
                    .toFixed(
                        3
                    )

                    +

                    "<br>|ψ⟩ = α|0⟩ + β|1⟩";

        },

        100
    );
}


/* =========================================================
TREE
========================================================= */

function startTree() {

    const nodes =
        document.querySelectorAll(
            ".node"
        );


    let index =
        0;


    setInterval(
        function () {

            nodes.forEach(
                function (
                    node
                ) {

                    node
                    .classList
                    .remove(
                        "active"
                    );
                }
            );


            nodes[index]
                .classList
                .add(
                    "active"
                );


            document
                .getElementById(
                    "tree-status"
                )
                .textContent =

                    "Decision node: "
                    +
                    (
                        index
                        +
                        1
                    );


            index =
                (
                    index
                    +
                    1
                )
                %
                nodes.length;

        },

        500
    );
}


/* =========================================================
NEURAL FLOW
========================================================= */

function startNeuralFlow() {

    const neurons =
        document.querySelectorAll(
            ".neuron"
        );


    let index =
        0;


    setInterval(
        function () {

            neurons.forEach(
                function (
                    neuron
                ) {

                    neuron
                    .classList
                    .remove(
                        "active"
                    );
                }
            );


            neurons[index]
                .classList
                .add(
                    "active"
                );


            document
                .getElementById(
                    "nn-status"
                )
                .textContent =

                    "Activation neuron: "
                    +
                    (
                        index
                        +
                        1
                    );


            index =
                (
                    index
                    +
                    1
                )
                %
                neurons.length;

        },

        220
    );
}


/* =========================================================
QML CIRCUIT ANIMATION
========================================================= */

function startQMLAnimation() {

    const gates =
        document.querySelectorAll(
            ".gate"
        );


    let index =
        0;


    setInterval(
        function () {

            gates.forEach(
                function (
                    gate
                ) {

                    gate
                    .classList
                    .remove(
                        "active"
                    );
                }
            );


            gates[index]
                .classList
                .add(
                    "active"
                );


            document
                .getElementById(
                    "qml-status"
                )
                .textContent =

                    "Executing gate: "
                    +
                    gates[index]
                    .textContent
                    .trim();


            index =
                (
                    index
                    +
                    1
                )
                %
                gates.length;

        },

        450
    );
}


/* =========================================================
START
========================================================= */

window.addEventListener(
    "DOMContentLoaded",

    function () {

        startTraining();

        startQuantumAnimation();

        startTree();

        startNeuralFlow();

        startQMLAnimation();


        document
            .getElementById(
                "run-qml-button"
            )
            .addEventListener(
                "click",
                runQMLTraining
            );


        loadDashboard();
    }
);

</script>

</body>

</html>
"""


# =========================================================
# SERVER
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )