import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs, make_moons, make_circles
import matplotlib
import time

# ======== 1. 基本设置 ========
# 中文字体设置
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 页面配置
st.set_page_config(page_title="聚类 vs 分类 实验平台", layout="wide")

# 居中大标题
st.markdown(
    """
    <h1 style="text-align:center; margin-bottom:20px; margin-top:-30px;">🔬 聚类 vs 分类 交互实验平台</h1>
    """,
    unsafe_allow_html=True
)

# ======== 2. 数据加载与缓存 ========
@st.cache_data
def load_datasets(random_seed):
    """加载所有预设的数据集"""
    X1, y1 = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=random_seed)
    X2, y2 = make_blobs(n_samples=300, centers=2, cluster_std=1.5, random_state=random_seed)
    X3, y3 = make_moons(n_samples=300, noise=0.15, random_state=random_seed)
    X4, y4 = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=random_seed)
    return [(X1, y1), (X2, y2), (X3, y3), (X4, y4)]

# ======== 3. 侧边栏交互组件 ========
st.sidebar.header("⚙️ 实验参数设置")
algo = st.sidebar.radio(
    "选择算法类型",
    options=["聚类", "分类"],
    captions=["探索数据内在结构", "根据已知数据预测新点"]
)

dataset_names = ["① 四簇球形", "② 两簇球形", "③ 月牙形", "④ 环形"]
dataset_choice = st.sidebar.selectbox(
    "选择数据集",
    options=list(range(len(dataset_names))),
    format_func=lambda x: dataset_names[x]
)

# 根据算法选择，滑块K的含义不同
if algo == "聚类":
    k = st.sidebar.slider("聚类数 K", 2, 8, 4)
else:
    k = st.sidebar.slider("KNN的邻居数 K", 1, 15, 3)

random_seed = st.sidebar.slider("随机种子", 0, 100, 42)

# 加载选定的数据集
datasets = load_datasets(random_seed)
X, y_true = datasets[dataset_choice]

# 分割数据
X_train, _, y_train, _ = train_test_split(X, y_true, test_size=0.1, random_state=random_seed)

# 仅在分类模式下显示“新数据点”调节器
if algo == "分类":
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 拖动滑块，改变新数据点位置")
    x_min, x_max = float(X[:, 0].min() - 1), float(X[:, 0].max() + 1)
    y_min, y_max = float(X[:, 1].min() - 1), float(X[:, 1].max() + 1)
    default_x = (x_min + x_max) / 2
    default_y = (y_min + y_max) / 2
    
    # 确保默认值在范围内
    if not x_min <= default_x <= x_max: default_x = x_min
    if not y_min <= default_y <= y_max: default_y = y_min

    new_x = st.sidebar.slider("新数据点 特征1 (X坐标)", x_min, x_max, default_x)
    new_y = st.sidebar.slider("新数据点 特征2 (Y坐标)", y_min, y_max, default_y)
    new_point = np.array([[new_x, new_y]])


# ======== 4. 主页面布局和绘图 ========
st.subheader(f"【{algo}】 - 算法结果展示")
col1, col2 = st.columns(2)

# ----- 左侧图：始终显示原始数据 -----
with col1:
    st.markdown("<h5 style='text-align: center;'>原始数据</h5>", unsafe_allow_html=True)
    fig_raw, ax_raw = plt.subplots(figsize=(6, 5))
    ax_raw.scatter(X[:, 0], X[:, 1], c='steelblue', alpha=0.6, s=30)
    ax_raw.set_title(f"数据集: {dataset_names[dataset_choice]}")
    ax_raw.set_xlabel("特征1")
    ax_raw.set_ylabel("特征2")
    ax_raw.axis('equal')
    st.pyplot(fig_raw)

# ----- 右侧图：根据算法显示不同结果 -----
with col2:
    if algo == "聚类":
        st.markdown(f"<h5 style='text-align: center;'>K-Means聚类结果 (K={k})</h5>", unsafe_allow_html=True)
        # 执行K-Means
        kmeans = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
        pred_labels = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_

        # 绘制聚类结果
        fig_res, ax_res = plt.subplots(figsize=(6, 5))
        ax_res.scatter(X[:, 0], X[:, 1], c=pred_labels, cmap='viridis', alpha=0.7, s=30)
        ax_res.scatter(centers[:, 0], centers[:, 1], c='red', marker='*', s=300, label='聚类中心')
        ax_res.set_xlabel("特征1")
        ax_res.set_ylabel("特征2")
        ax_res.legend()
        ax_res.axis('equal')
        st.pyplot(fig_res)
        st.markdown(f"<p style='text-align: center;'>K-Means算法将无标签数据自动分为了 {k} 个簇。</p>", unsafe_allow_html=True)

    else: # algo == "分类"
        st.markdown(f"<h5 style='text-align: center;'>KNN分类过程演示 (K={k})</h5>", unsafe_allow_html=True)
        plot_placeholder = st.empty()
        caption_placeholder = st.empty()

        # 训练KNN模型
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        # 找到邻居并预测
        distances, indices = knn.kneighbors(new_point)
        neighbors = X_train[indices[0]]
        pred_label = knn.predict(new_point)[0]
        
        # 获取颜色映射
        cmap = plt.get_cmap('viridis')
        unique_labels = np.unique(y_train)
        vmin, vmax = min(unique_labels), max(unique_labels)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        # --- 第一阶段：显示决策过程（黑星） ---
        fig_before, ax_before = plt.subplots(figsize=(6, 5))
        ax_before.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, alpha=0.3, s=30, vmin=vmin, vmax=vmax, label='已知数据点')
        ax_before.scatter(new_point[0, 0], new_point[0, 1], color='black', marker='*', s=350, zorder=10, label='新数据点') # 使用 color='black'
        ax_before.scatter(neighbors[:, 0], neighbors[:, 1], s=150, facecolors='none', edgecolors='red', linewidths=2, zorder=9, label=f'最近的{k}个邻居')
        for neighbor in neighbors:
            ax_before.plot([new_point[0, 0], neighbor[0]], [new_point[0, 1], neighbor[1]], 'k--', alpha=0.5)
        
        ax_before.set_title(f"正在分析新数据点...")
        ax_before.set_xlabel("特征1")
        ax_before.set_ylabel("特征2")
        ax_before.legend(loc='best')
        ax_before.axis('equal')
        plot_placeholder.pyplot(fig_before)
        caption_placeholder.markdown("<p style='text-align: center; font-style: italic;'>正在根据邻居投票...</p>", unsafe_allow_html=True)

        time.sleep(1.5)

        # --- 第二阶段：显示分类结果（彩色星） ---
        pred_color = cmap(norm(pred_label)) # 获取预测类别对应的具体颜色

        fig_after, ax_after = plt.subplots(figsize=(6, 5))
        ax_after.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap, alpha=0.3, s=30, vmin=vmin, vmax=vmax, label='已知数据点')
        # --- 核心修改：用正确的颜色和标签绘制星星，以生成正确的图例 ---
        ax_after.scatter(
            new_point[0, 0], new_point[0, 1], 
            color=pred_color, # 直接使用计算好的颜色
            marker='*', s=350, zorder=10, 
            edgecolors='black', linewidth=0.5, 
            label=f'预测结果 (类别 {pred_label})' # 这个label会关联上面正确的颜色
        )
        ax_after.scatter(neighbors[:, 0], neighbors[:, 1], s=150, facecolors='none', edgecolors='red', linewidths=2, zorder=9, label=f'最近的{k}个邻居')

        ax_after.set_title(f"预测结果：新点属于类别 {pred_label}")
        ax_after.set_xlabel("特征1")
        ax_after.set_ylabel("特征2")
        ax_after.legend(loc='best')
        ax_after.axis('equal')

        plot_placeholder.pyplot(fig_after)
        
        color_note = f"(注：紫色点为类别0，黄色点为类别1)" if len(unique_labels) == 2 else ""
        caption_placeholder.markdown(f"<p style='text-align: center;'>根据最近的 {k} 个邻居的类别投票，新点被分类为 <b>类别 {pred_label}</b>。{color_note}</p>", unsafe_allow_html=True)