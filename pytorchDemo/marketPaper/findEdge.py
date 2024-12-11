## 基于已有node、edge推断潜在的edge
## 方案1：source_service
## 方案2：使用协同过滤推断新边

import networkx as nx
import ssl
print(ssl.OPENSSL_VERSION)
# 构造图
G = nx.DiGraph()

# 添加节点
nodes = [
    {"id": "web节点1", "ip": "192.168.1.1", "机房": "北京", "网络区域": "区域A", "微服务": "web工程"},
    {"id": "web节点2", "ip": "192.168.1.2", "机房": "北京", "网络区域": "区域A", "微服务": "web工程"},
    {"id": "web节点3", "ip": "192.169.1.1", "机房": "深圳", "网络区域": "区域A", "微服务": "web工程"},
    {"id": "web节点4", "ip": "192.169.1.2", "机房": "深圳", "网络区域": "区域A", "微服务": "web工程"},
    {"id": "service节点1", "ip": "192.168.1.3", "机房": "北京", "网络区域": "区域A", "微服务": "service工程"},
    {"id": "service节点2", "ip": "192.168.1.4", "机房": "北京", "网络区域": "区域A", "微服务": "service工程"},
    {"id": "service节点3", "ip": "192.169.1.3", "机房": "深圳", "网络区域": "区域A", "微服务": "service工程"},
    {"id": "service节点4", "ip": "192.169.1.4", "机房": "深圳", "网络区域": "区域A", "微服务": "service工程"},
    {"id": "kafka节点1", "ip": "192.168.1.5", "机房": "北京", "网络区域": "区域A", "微服务": "kafka"},
    {"id": "kafka节点2", "ip": "192.168.1.6", "机房": "北京", "网络区域": "区域A", "微服务": "kafka"},
    {"id": "kafka节点3", "ip": "192.168.1.7", "机房": "北京", "网络区域": "区域A", "微服务": "kafka"},
    {"id": "kafka节点4", "ip": "192.169.1.5", "机房": "深圳", "网络区域": "区域A", "微服务": "kafka"},
    {"id": "kafka节点5", "ip": "192.169.1.6", "机房": "深圳", "网络区域": "区域A", "微服务": "kafka"},
    {"id": "kafka节点6", "ip": "192.169.1.7", "机房": "深圳", "网络区域": "区域A", "微服务": "kafka"},
    {"id": "mysql节点1", "ip": "192.168.1.8", "机房": "北京", "网络区域": "区域A", "微服务": "MGR"},
    {"id": "mysql节点2", "ip": "192.168.1.9", "机房": "北京", "网络区域": "区域A", "微服务": "MGR"},
    {"id": "mysql节点3", "ip": "192.168.1.10", "机房": "北京", "网络区域": "区域A", "微服务": "MGR"},
    {"id": "mysql节点4", "ip": "192.168.1.11", "机房": "北京", "网络区域": "区域A", "微服务": "MGR"},
    {"id": "mysql节点5", "ip": "192.169.1.8", "机房": "深圳", "网络区域": "区域A", "微服务": "MGR"},
    {"id": "mysql节点6", "ip": "192.169.1.9", "机房": "深圳", "网络区域": "区域A", "微服务": "MGR"},
    {"id": "mysql节点7", "ip": "192.169.1.10", "机房": "深圳", "网络区域": "区域A", "微服务": "MGR"},
    {"id": "mysql节点8", "ip": "192.169.1.11", "机房": "深圳", "网络区域": "区域A", "微服务": "MGR"},
    {"id": "order节点1", "ip": "192.168.1.12", "机房": "北京", "网络区域": "区域A", "微服务": "order工程"},
    {"id": "order节点2", "ip": "192.168.1.13", "机房": "北京", "网络区域": "区域A", "微服务": "order工程"},
    {"id": "order节点3", "ip": "192.168.1.12", "机房": "深圳", "网络区域": "区域A", "微服务": "order工程"},
    {"id": "order节点4", "ip": "10.128.1.13", "机房": "深圳", "网络区域": "区域B", "微服务": "order工程"}
]
# 添加已知边
edges = [
    ("web节点2", "service节点1", "TCP:80", "短连接"),
    ("service节点2", "web节点2", "TCP:80", "短连接"),
    ("service节点4", "kafka节点4", "TCP:9092", "短连接"),
    ("order节点2", "kafka节点2", "TCP:9092", "短连接"),
    ("service节点2", "kafka节点2", "TCP:9092", "短连接"),
    ("order节点3", "kafka节点4", "TCP:9092", "短连接"),
    ("service节点3", "mysql节点5", "TCP:3306", "短连接"),
    ("order节点2", "mysql节点2", "TCP:3306", "短连接"),
    ("service节点2", "mysql节点2", "TCP:3306", "短连接"),
    ("web节点3", "service节点3", "TCP:80", "短连接"),
    ("order节点3", "mysql节点6", "TCP:3306", "短连接")
]

for node in nodes:
    G.add_node(node["id"], **node)

for edge in edges:
    for node in nodes:
        if (node["id"] == edge[0]):
            source_node = node
        if (node["id"] == edge[1]):
            target_node = node

    print(source_node)
    print(target_node)
    G.add_edge(edge[0], edge[1], s_微服务=source_node[4], s_网络=source_node[3], s_机房=source_node[2], 协议=edge[2], 连接类型=edge[3])

    #TODO 策略筛选保留有效策略
    #TODO 策略两端至少有一端归属到合理存在的微服务/集群中

# 查找潜在的边。
# 复杂度：O(E*N*N)
# 1.轮询所有边
# 2.轮询所有节点作为源端
# 3.轮询所有节点作为目的端
# def find_potential_edges(graph):
#     potential_edges = set()
#     for source, target, data in graph.edges(data=True):
#         source_service = graph.nodes[source]["微服务"]
#         target_service = graph.nodes[target]["微服务"]
#         source_area = graph.nodes[source]["网络区域"]
#         target_area = graph.nodes[target]["网络区域"]
#         source_room = graph.nodes[source]["机房"]
#         target_room = graph.nodes[target]["机房"]
#
#         # 找到同样微服务，网络区域和机房的其他节点对
#         for s in graph.nodes:
#             for t in graph.nodes:
#                 if(s != t):
#                     if (graph.nodes[s]["微服务"] == source_service and
#                             graph.nodes[t]["微服务"] == target_service and
#                             graph.nodes[s]["网络区域"] == source_area and
#                             graph.nodes[t]["网络区域"] == target_area and
#                             graph.nodes[s]["机房"] == source_room and
#                             graph.nodes[t]["机房"] == target_room):
#                         if not graph.has_edge(s, t):
#                             #yield s, t, data["协议"], data["连接类型"]
#                             potential_edges.add((s, t, data["协议"], data["连接类型"]))
#     return potential_edges


# 查找潜在的边。
# 复杂度：O(E*2*N)
# 1.轮询所有边
# 2.分别查找源端节点列表sList、目的端节点列表tList

# 对所有边进行排序
def edge_key(edge):
    source, target, _ = edge
    return (max(source, target), min(source, target))
def find_potential_edges(graph):
    potential_edges = set()
    source_nodes = set()
    target_nodes = set()

    sorted_edges = sorted(G.edges(data=True), key=edge_key)
    print("\n排序边:")
    for edge in sorted_edges:
        print(edge)

    for source, target, data in graph.edges(data=True):
        source_service = graph.nodes[source]["微服务"]
        target_service = graph.nodes[target]["微服务"]
        source_area = graph.nodes[source]["网络区域"]
        target_area = graph.nodes[target]["网络区域"]
        source_room = graph.nodes[source]["机房"]
        target_room = graph.nodes[target]["机房"]


        # 找到同样微服务，网络区域和机房的节点列表
        for node in graph.nodes:
            if (graph.nodes[node]["微服务"] == source_service and
                graph.nodes[node]["网络区域"] == source_area and
                graph.nodes[node]["机房"] == source_room):
                source_nodes.add(node)
            if (graph.nodes[node]["微服务"] == target_service and
                graph.nodes[node]["网络区域"] == target_area and
                graph.nodes[node]["机房"] == target_room):
                target_nodes.add(node)


        # 找到同样微服务，网络区域和机房的其他节点对
        for s in graph.nodes:
            for t in graph.nodes:
                if(s != t):
                    if (graph.nodes[s]["微服务"] == source_service and
                            graph.nodes[t]["微服务"] == target_service and
                            graph.nodes[s]["网络区域"] == source_area and
                            graph.nodes[t]["网络区域"] == target_area and
                            graph.nodes[s]["机房"] == source_room and
                            graph.nodes[t]["机房"] == target_room):
                        if not graph.has_edge(s, t):
                            #yield s, t, data["协议"], data["连接类型"]
                            potential_edges.add((s, t, data["协议"], data["连接类型"]))
    return potential_edges

# 推断 web节点1 访问 service节点2
#new_edges = set()
#new_edges.update(propagate_edges(G, "web节点1", "service工程"))

# 获取潜在的边
new_edges = find_potential_edges(G)

for edge in new_edges:
    G.add_edge(edge[0], edge[1], 协议=edge[2], 连接类型=edge[3])

# 对所有边进行排序
sorted_edges = sorted(G.edges(data=True), key=lambda x: (x[0], x[1]))

G.clear_edges()
G.add_edges_from(sorted_edges)

# 输出所有边
print("\n所有边:")
for edge in G.edges(data=True):
    print(edge)