import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import pulp as pp
from sklearn import datasets
from sklearn.cluster import KMeans


def create_data(A, M, V, state):
    state = state or {}
    A = int(A)
    M = int(M)
    V = int(V)
    x, y = datasets.make_classification(n_samples=A * M,
                                        n_features=V,
                                        n_informative=V,
                                        n_redundant=0,
                                        n_repeated=0,
                                        n_classes=A,
                                        n_clusters_per_class=1,
                                        class_sep=3,
                                        flip_y=0,
                                        shuffle=False,
                                        scale=1000,
                                        )
    x = np.array(x, dtype=int)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y)
    state["before_image"] = fig
    state["x"] = x
    return x, fig, state


def cluster(A, M, V, state):
    state = state or {}
    state["before_image"] = state["before_image"] or plt.figure()
    A = int(A)
    M = int(M)
    V = int(V)
    x = state["x"]
    model = KMeans(n_clusters=A, n_init='auto')
    model.fit(x)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=model.labels_)
    ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], model.cluster_centers_[:, 2],
               c=np.arange(A), marker='x', s=100)
    state["after_image"] = fig
    state["model"] = model

    return state["before_image"], fig, state


def predict(A, M, V, state):
    state = state or {}
    A = int(A)
    M = int(M)
    V = int(V)
    x = state["x"]
    model = state["model"]
    y = model.predict(x)
    state["y"] = y
    summary = ""
    for i in range(A):
        summary += f"第{i}类有{np.count_nonzero(y == i)}个\n"
    return np.concatenate((x, y.reshape(-1, 1)), axis=1), summary, state


def generate_plans(A, M, V, state):
    state = state or {}
    A = int(A)
    M = int(M)
    V = int(V)
    plans = np.random.randint(0, 3, (A * 3, V + 1))
    # plans = np.diag(np.ones(A, dtype=int))
    # plans = np.concatenate((plans, np.ones((A, 1), dtype=int)), axis=1)
    state["plans"] = plans
    return plans, state


def calculate_best(A, M, V, state):
    state = state or {}
    A = int(A)
    M = int(M)
    V = int(V)
    plans = state["plans"]
    y = state["y"]
    count_y = np.zeros(A)
    for i in range(A):
        count_y[i] = np.count_nonzero(y == i)

    mylp = pp.LpProblem("solver", pp.LpMaximize)
    xx = [pp.LpVariable(f"x{i}", 0, None, pp.LpInteger) for i in range(plans.shape[0])]
    mylp += pp.lpSum([plans[i, -1] * xx[i] for i in range(plans.shape[0])])
    for i in range(V):
        mylp += pp.lpSum([plans[j, i] * xx[j] for j in range(plans.shape[0])]) <= count_y[i]
    # print(mylp)
    mylp.solve()
    return f"Result: x={[v.varValue for v in mylp.variables()]}\ntarget={mylp.objective}={pp.value(mylp.objective)}\n\n" \
           f"Detail: {mylp}", state


with gr.Blocks() as demo:
    A = gr.Number(label="种类数", value=5)
    M = gr.Number(label="每种的人数", value=4)
    V = gr.Number(label="特征数", value=3)
    state = gr.State()
    with gr.Tab("Generate Data"):
        create_data_button = gr.Button("Generate")
        with gr.Row():
            csv_output = gr.DataFrame(headers=[str(_) for _ in range(int(V.value))], datatype="number",
                                      col_count=int(V.value))
            image_output = gr.Plot()

    with gr.Tab("Cluster"):
        cluster_button = gr.Button("Cluster")
        with gr.Row():
            before_image = gr.Plot()
            after_image = gr.Plot()
    with gr.Tab("Predict"):
        predict_button = gr.Button("Predict")
        with gr.Row():
            predict_output = gr.DataFrame(headers=(["x" + str(_) for _ in range(int(V.value))] + ["y"]),
                                          datatype="number", col_count=int(V.value + 1))
            summary_output = gr.Textbox()
    with gr.Tab("Group"):
        generate_plans_button = gr.Button("Generate Plans")
        calculate_best_button = gr.Button("Calculate Best")
        with gr.Row():
            group_input = gr.DataFrame(headers=(["第" + str(_) + "类需要人数" for _ in range(int(V.value))] + ["得分"]))
            group_output = gr.Textbox()

    create_data_button.click(create_data, inputs=[A, M, V, state], outputs=[csv_output, image_output, state])
    cluster_button.click(cluster, inputs=[A, M, V, state], outputs=[before_image, after_image, state])
    predict_button.click(predict, inputs=[A, M, V, state], outputs=[predict_output, summary_output, state])
    generate_plans_button.click(generate_plans, inputs=[A, M, V, state], outputs=[group_input, state])
    calculate_best_button.click(calculate_best, inputs=[A, M, V, state], outputs=[group_output, state])

demo.launch()
