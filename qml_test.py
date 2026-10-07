from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix

from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit_algorithms.optimizers import COBYLA

import json


# =========================================================
# TRAINING HISTORY
# =========================================================

training_history = []


# =========================================================
# CUSTOM COBYLA RECORDER
# =========================================================

class RecordingCOBYLA(COBYLA):

    def minimize(
        self,
        fun,
        x0,
        jac=None,
        bounds=None,
    ):

        def recorded_fun(x):

            value = fun(x)

            value_float = float(value)

            training_history.append(
                value_float
            )

            step = len(
                training_history
            )

            print(
                f"VQC optimization step {step:03d} "
                f"| objective = {value_float:.6f}"
            )

            return value

        return super().minimize(
            recorded_fun,
            x0,
            jac=jac,
            bounds=bounds,
        )


# =========================================================
# DATA
# =========================================================

iris = load_iris()

X = iris.data[:100, 2:4]
y = iris.target[:100]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)


# =========================================================
# SCALE TO QUANTUM ANGLES
# =========================================================

scaler = MinMaxScaler(
    feature_range=(
        0,
        3.141592653589793,
    )
)

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)


# =========================================================
# QUANTUM MODEL
# =========================================================

num_qubits = 2


feature_map = zz_feature_map(
    feature_dimension=num_qubits,
    reps=2,
)


ansatz = real_amplitudes(
    num_qubits=num_qubits,
    reps=2,
)


optimizer = RecordingCOBYLA(
    maxiter=150
)


vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
)


# =========================================================
# TRAIN
# =========================================================

print()

print(
    "==================================="
)

print(
    " TRAINING VARIATIONAL QUANTUM CLASSIFIER"
)

print(
    "==================================="
)

print()


vqc.fit(
    X_train,
    y_train,
)


# =========================================================
# TEST
# =========================================================

predictions = vqc.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    predictions,
)


cm = confusion_matrix(
    y_test,
    predictions,
)


# =========================================================
# FIT RESULT
# =========================================================

fit_result = getattr(
    vqc,
    "fit_result",
    None,
)


optimizer_iterations = len(
    training_history
)


final_objective = None


if len(training_history) > 0:

    final_objective = float(
        training_history[-1]
    )


if fit_result is not None:

    if hasattr(
        fit_result,
        "fun",
    ):

        try:

            final_objective = float(
                fit_result.fun
            )

        except Exception:

            pass


# =========================================================
# SAVE RESULT
# =========================================================

result_data = {

    "model":
        "Variational Quantum Classifier",

    "features": [
        "petal length",
        "petal width",
    ],

    "qubits":
        num_qubits,

    "feature_map":
        "ZZFeatureMap reps=2",

    "ansatz":
        "RealAmplitudes reps=2",

    "optimizer":
        "COBYLA maxiter=150",

    "train_samples":
        int(
            len(X_train)
        ),

    "test_samples":
        int(
            len(X_test)
        ),

    "accuracy":
        float(
            accuracy
        ),

    "confusion_matrix":
        cm.astype(
            int
        ).tolist(),

    "training_history":
        [
            float(x)
            for x in training_history
        ],

    "iterations":
        int(
            optimizer_iterations
        ),

    "final_objective":
        final_objective,
}


with open(
    "qml_result.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        result_data,
        file,
        indent=4,
    )


# =========================================================
# FINAL RESULT
# =========================================================

print()

print(
    "==================================="
)

print(
    " OPTIMIZED QUANTUM ML RESULT"
)

print(
    "==================================="
)


print(
    "Model: Variational Quantum Classifier"
)

print(
    "Features: petal length + petal width"
)

print(
    "Qubits:",
    num_qubits,
)

print(
    "Feature Map: ZZFeatureMap reps=2"
)

print(
    "Ansatz: RealAmplitudes reps=2"
)

print(
    "Optimizer: COBYLA maxiter=150"
)

print(
    "Train samples:",
    len(X_train),
)

print(
    "Test samples:",
    len(X_test),
)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print(
    "Optimization evaluations:",
    optimizer_iterations,
)


if final_objective is not None:

    print(
        f"Final objective: {final_objective:.6f}"
    )


print()

print(
    "Confusion Matrix:"
)

print(
    cm
)

print()

print(
    "Saved: qml_result.json"
)

print(
    "==================================="
)