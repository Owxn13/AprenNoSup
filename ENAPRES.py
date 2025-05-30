import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

print("=== ANÁLISIS CLUSTERING Y CLASIFICACIÓN - ENAPRES 2023 ===")

#1. CARGA Y PREPARACIÓN DE DATOS
df = pd.read_csv(r'C:\Users\HP\Downloads\ENAPRES.csv', sep=';')

print(f" Dataset cargado: {df.shape}")
print(f"Columnas: {list(df.columns)}")

#Conversión a numérico
for col in df.columns:
    if col not in ['ANIO']:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            pass

print(" Conversión a numérico completada")

#Seleccionar variables para clustering
variables_clustering = []
for col in df.columns:
    if df[col].dtype in ['int64', 'float64'] and df[col].nunique() > 1:
        variables_clustering.append(col)

print(f"Variables para clustering: {variables_clustering}")

X = df[variables_clustering].copy()
if X.isnull().sum().sum() > 0:
    print("Rellenando valores faltantes...")
    X = X.fillna(X.mean())

print(f"Dataset para clustering: {X.shape}")

#Estandarización
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(" Datos estandarizados")

#2. DETERMINACIÓN ÓPTIMA DE CLUSTERS (Múltiples métodos)
print("\n=== DETERMINACIÓN ÓPTIMA DE CLUSTERS ===")

silhouette_scores = []
calinski_scores = []
davies_bouldin_scores = []
n_clusters_range = range(2, 11)

print("Evaluando diferentes números de clusters...")
for n_clust in n_clusters_range:
    hierarchical = AgglomerativeClustering(n_clusters=n_clust, linkage='ward')
    cluster_labels_temp = hierarchical.fit_predict(X_scaled)
    
    sil_score = silhouette_score(X_scaled, cluster_labels_temp)
    ch_score = calinski_harabasz_score(X_scaled, cluster_labels_temp)
    db_score = davies_bouldin_score(X_scaled, cluster_labels_temp)
    
    silhouette_scores.append(sil_score)
    calinski_scores.append(ch_score)
    davies_bouldin_scores.append(db_score)
    
    print(f"k={n_clust}: Silhouette={sil_score:.3f}, Calinski-Harabasz={ch_score:.1f}, Davies-Bouldin={db_score:.3f}")

#Gap Statistic (método académico) - VERSIÓN CORREGIDA
def gap_statistic(X_data, max_clusters=8, n_refs=10):
    gaps = []
    k_values = []
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_data)
        inertia_real = kmeans.inertia_
        
        inertias_random = []
        for _ in range(n_refs):
            random_data = np.random.uniform(X_data.min(), X_data.max(), X_data.shape)
            kmeans_random = KMeans(n_clusters=k, random_state=42, n_init=10)
            inertias_random.append(kmeans_random.fit(random_data).inertia_)
        
        gap = np.log(np.mean(inertias_random)) - np.log(inertia_real)
        gaps.append(gap)
        k_values.append(k)
    
    return gaps, k_values

print("\nCalculando Gap Statistic...")
gaps, k_values_gap = gap_statistic(X_scaled)
best_n_clusters_gap = k_values_gap[np.argmax(gaps)]

#Mejor número de clusters
best_n_clusters = n_clusters_range[np.argmax(silhouette_scores)]
best_silhouette = max(silhouette_scores)

print(f"\n NÚMERO ÓPTIMO DE CLUSTERS:")
print(f"  • Silhouette Score: k={best_n_clusters} (score: {best_silhouette:.3f})")
print(f"  • Gap Statistic: k={best_n_clusters_gap} (gap: {max(gaps):.3f})")
print(f"  • Decisión final: k={best_n_clusters}")

#3. COMPARACIÓN DE ALGORITMOS
print(f"\n=== COMPARACIÓN DE ALGORITMOS ===")

algorithms = {
    'Ward Hierarchical': AgglomerativeClustering(n_clusters=best_n_clusters, linkage='ward'),
    'K-Means': KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10),
    'Complete Linkage': AgglomerativeClustering(n_clusters=best_n_clusters, linkage='complete'),
}

algorithm_results = {}
cluster_assignments = {}

print("Evaluando algoritmos...")
for name, algorithm in algorithms.items():
    labels = algorithm.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels)
    algorithm_results[name] = sil_score
    cluster_assignments[name] = labels
    print(f"{name}: Silhouette = {sil_score:.3f}")

best_algorithm = max(algorithm_results, key=algorithm_results.get)
print(f"\n Mejor algoritmo: {best_algorithm} (Silhouette: {algorithm_results[best_algorithm]:.3f})")

#4. CLUSTERING FINAL
print(f"\n=== CLUSTERING FINAL ===")
cluster_labels = cluster_assignments[best_algorithm]

unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
print(f"Distribución de clusters:")
for cluster_id, count in zip(unique_clusters, counts):
    print(f"  Cluster {cluster_id}: {count} muestras ({count/len(cluster_labels)*100:.1f}%)")

#5. ANÁLISIS DE FEATURE IMPORTANCE
print(f"\n=== IMPORTANCIA DE VARIABLES ===")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, cluster_labels)

feature_importance = pd.DataFrame({
    'Variable': variables_clustering,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("Ranking de importancia:")
for idx, row in feature_importance.iterrows():
    print(f"  {row['Variable']}: {row['Importance']:.3f}")

#6. CLASIFICACIÓN KNN
print(f"\n=== CLASIFICACIÓN KNN ===")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, cluster_labels, test_size=0.3, random_state=42, stratify=cluster_labels
)

k_range = range(1, min(21, len(X_train)//5))
k_scores = []

print("Optimizando K para KNN...")
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

best_k = k_range[np.argmax(k_scores)]
best_knn_score = max(k_scores)

knn_final = KNeighborsClassifier(n_neighbors=best_k)
knn_final.fit(X_train, y_train)
y_pred = knn_final.predict(X_test)

print(f" Mejor K: {best_k}, Accuracy: {best_knn_score:.3f}")

#7. PCA PARA VISUALIZACIÓN
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f" PCA aplicado - Varianza explicada: {sum(pca.explained_variance_ratio_):.1%}")

#8. VISUALIZACIONES PRINCIPALES

#8.1 Silhouette Score
plt.figure(figsize=(10, 6))
plt.plot(n_clusters_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
plt.axvline(x=best_n_clusters, color='red', linestyle='--', alpha=0.7)
plt.title('Silhouette Score para Selección Óptima de Clusters', fontsize=14, fontweight='bold')
plt.xlabel('Número de Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#8.2 Gap Statistic  
plt.figure(figsize=(10, 6))
plt.plot(k_values_gap, gaps, 'go-', linewidth=2, markersize=8)
plt.axvline(x=best_n_clusters_gap, color='red', linestyle='--', alpha=0.7)
plt.title('Gap Statistic para Determinación de Clusters', fontsize=14, fontweight='bold')
plt.xlabel('Número de Clusters')
plt.ylabel('Gap Statistic')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#8.3 Comparación de Algoritmos
plt.figure(figsize=(10, 6))
algorithms_names = list(algorithm_results.keys())
algorithms_scores = list(algorithm_results.values())
colors = ['red' if name == best_algorithm else 'lightblue' for name in algorithms_names]
bars = plt.bar(range(len(algorithms_names)), algorithms_scores, color=colors)

#Añadir valores en las barras
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.3f}', ha='center', va='bottom', fontsize=10)

plt.title('Comparación de Algoritmos de Clustering', fontsize=14, fontweight='bold')
plt.xlabel('Algoritmo')
plt.ylabel('Silhouette Score')
plt.xticks(range(len(algorithms_names)), [name.split()[0] for name in algorithms_names])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#8.2 Dendrograma
plt.figure(figsize=(12, 8))
linkage_matrix = linkage(X_scaled, method='ward')
dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.title('Dendrograma - Clustering Jerárquico Ward', fontsize=16, fontweight='bold')
plt.xlabel('Índice de Muestras')
plt.ylabel('Distancia Ward')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#8.3 Clusters en PCA
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                     cmap='viridis', alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
plt.title(f'Clusters Finales - {best_algorithm}\n{best_n_clusters} clusters identificados', 
          fontsize=16, fontweight='bold')
plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]:.1%} de varianza)', fontsize=12)
plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]:.1%} de varianza)', fontsize=12)
cbar = plt.colorbar(scatter)
cbar.set_label('Cluster', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#8.4 Selección K para KNN
plt.figure(figsize=(10, 6))
plt.plot(k_range, k_scores, 'go-', linewidth=3, markersize=8)
plt.axvline(x=best_k, color='red', linestyle='--', linewidth=2,
            label=f'Mejor K={best_k} (Accuracy={best_knn_score:.3f})')
plt.title('Selección del Valor Óptimo de K para KNN', fontsize=16, fontweight='bold')
plt.xlabel('Valor de K (Número de Vecinos)')
plt.ylabel('Accuracy (Validación Cruzada)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#8.5 Matriz de confusión
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, 
            cbar_kws={'label': 'Número de Muestras'})
plt.title('Matriz de Confusión - Clasificación KNN', fontsize=16, fontweight='bold')
plt.ylabel('Cluster Real')
plt.xlabel('Cluster Predicho')
plt.tight_layout()
plt.show()

#8.6 Feature Importance
plt.figure(figsize=(10, 6))
bars = plt.barh(feature_importance['Variable'], feature_importance['Importance'],
                color=plt.cm.viridis(np.linspace(0.2, 0.8, len(feature_importance))))
plt.title('Importancia de Variables para Clustering\n(Random Forest Feature Importance)', 
          fontsize=16, fontweight='bold')
plt.xlabel('Importancia Relativa')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

#8.7 Distribución de clusters
plt.figure(figsize=(10, 6))
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_counts)))
bars = plt.bar(cluster_counts.index, cluster_counts.values, color=colors,
               edgecolor='black', linewidth=1)

plt.title('Distribución de Muestras por Cluster', fontsize=16, fontweight='bold')
plt.xlabel('Cluster')
plt.ylabel('Número de Muestras')

#Etiquetas en barras
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{int(height)}\n({height/len(cluster_labels)*100:.1f}%)', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

#9. CARACTERIZACIÓN DE CLUSTERS
print(f"\n=== CARACTERIZACIÓN DE CLUSTERS ===")

df_results = df.copy()
df_results['Cluster'] = cluster_labels
df_results['PC1'] = X_pca[:, 0]
df_results['PC2'] = X_pca[:, 1]

for cluster_id in sorted(df_results['Cluster'].unique()):
    cluster_data = df_results[df_results['Cluster'] == cluster_id]
    print(f"\n--- CLUSTER {cluster_id} ({len(cluster_data)} muestras, {len(cluster_data)/len(df_results)*100:.1f}%) ---")
    
    for col in variables_clustering:
        mean_val = cluster_data[col].mean()
        std_val = cluster_data[col].std()
        print(f"  {col}: μ={mean_val:.2f} ± {std_val:.2f}")

#10. ANÁLISIS DE VECINOS MÁS CERCANOS
print(f"\n=== ANÁLISIS DE VECINOS MÁS CERCANOS ===")

def analizar_vecinos_cercanos(muestra_idx, k=5):
    """Analiza los k vecinos más cercanos de una muestra específica"""
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto')
    nbrs.fit(X_scaled)
    
    distancias, indices = nbrs.kneighbors([X_scaled[muestra_idx]])
    distancias = distancias[0][1:]  # Excluir la misma muestra
    indices = indices[0][1:]
    
    print(f"\n ANÁLISIS DE VECINOS - Muestra {muestra_idx}")
    print(f"Cluster asignado: {cluster_labels[muestra_idx]}")
    print("Vecinos más cercanos:")
    for i, (dist, vecino_idx) in enumerate(zip(distancias, indices)):
        print(f"  {i+1}. Índice {vecino_idx}: distancia={dist:.3f}, cluster={cluster_labels[vecino_idx]}")

#Ejemplo de análisis
if len(X) > 0:
    analizar_vecinos_cercanos(0, k=5)

#11. REPORTE FINAL
print(f"\n" + "="*70)
print(" REPORTE FINAL DEL ANÁLISIS")
print("="*70)

print(f"\n METODOLOGÍA:")
print(f"  • Dataset: {df.shape[0]} registros, {len(variables_clustering)} variables")
print(f"  • Variables analizadas: {', '.join(variables_clustering)}")
print(f"  • Preprocesamiento: Estandarización Z-score")
print(f"  • Métodos de validación: Silhouette, Calinski-Harabasz, Davies-Bouldin, Gap Statistic")

print(f"\n RESULTADOS CLUSTERING:")
print(f"  • Número óptimo de clusters: {best_n_clusters}")
print(f"  • Mejor algoritmo: {best_algorithm}")
print(f"  • Silhouette Score: {best_silhouette:.3f}")
print(f"  • Variable más importante: {feature_importance.iloc[0]['Variable']}")

print(f"\n CLASIFICACIÓN KNN:")
print(f"  • Mejor K: {best_k}")
print(f"  • Accuracy (validación cruzada): {best_knn_score:.3f}")
print(f"  • Accuracy (conjunto de prueba): {knn_final.score(X_test, y_test):.3f}")

print(f"\n VISUALIZACIÓN:")
print(f"  • Varianza explicada PCA (2D): {sum(pca.explained_variance_ratio_):.1%}")
print(f"  • {len(unique_clusters)} gráficos generados")

print(f"\n--- Reporte de Clasificación KNN Detallado ---")
print(classification_report(y_test, y_pred))