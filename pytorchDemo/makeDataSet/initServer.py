import random
import pandas as pd
from collections import defaultdict


# 配置系统映射
def configure_mappings():
    systems_services_map = {
        "APP": ["web", "manage", "service", "MGR", "Nginx", "ES", "kafka", "ha", "web5", "agent", "mos", "android"],
        "ACT": ["app", "core", "jifen", "quanyi", "bill", "fenqi", "batch", "huankuan", "detail", "user", "MGR", "ha",
                "Nginx", "Redis", "ES", "address"],
        "PAY": ["web", "app", "NUCC", "JCB", "AE", "VISA", "MASTER", "UnionPAY", "MGR", "ha", "Nginx", "Redis"],
        "SHOP": ["web", "cart", "mcc", "price", "product", "city", "address", "app", "MGR", "ha", "Nginx", "Redis"],
        "EMS": ["web", "app", "MGR", "ha", "Nginx", "Redis", "address", "user", "order", "detail"],
        "DCOS": ["web", "compute", "net", "story", "security", "backup", "subnet", "vpc", "bms", "sfs", "lvm", "s3",
                 "service", "order", "adapter", "MGR", "ha", "Kafka", "Nginx"],
        "MAIL": ["web", "mail", "gateway", "MGR", "ha", "Kafka", "Nginx"],
        "MSG": ["web", "msg", "gateway", "MGR", "ha", "Kafka", "Nginx"],
        "CHAT": ["web", "log", "image", "voice", "txt", "chat", "gateway", "robt", "msg", "MGR", "ha", "Kafka",
                 "Nginx"],
        "CALLCENTER": ["web", "gateway", "service", "CTCC", "CMCC", "CUCC", "MGR", "ha", "Kafka", "Nginx"],
        "OA": ["web", "jbpm", "apply", "approval", "news", "ehr", "salary", "contract", "msg", "gateway", "MGR", "ha",
               "Kafka", "Nginx", "Kafka"],
        "CMDB": ["web", "compute", "net", "story", "security", "backup", "app", "ES", "MGR", "ha", "Nginx", "adapter"],
        "NATO": ["web", "zabbix", "exporter", "skywalking", "apm", "app", "MGR", "ha", "ES", "Nginx"],
        "CHGM": ["web", "service", "MGR", "ha"],
        "YUM": ["web", "yum", "MGR", "ha"],
        "HABOR": ["web", "habor", "MGR", "ha"],
        "DOCKER": ["web", "docker", "MGR", "ha"],
        "LOG": ["web", "hot", "cold", "log", "ES", "DORIS", "MGR", "ha"],
        "JENKINS": ["web", "pipline", "jenkins", "MGR", "ha"],
    }
    system_level_map = {
        "APP": "A+",
        "ACT": "A+",
        "PAY": "A",
        "SHOP": "A",
        "EMS": "A",
        "DCOS": "B",
        "MAIL": "B",
        "MSG": "B",
        "CHAT": "B",
        "CALLCENTER": "B",
        "OA": "B",
        "CMDB": "C",
        "NATO": "C",
        "CHGM": "C",
        "YUM": "C",
        "HABOR": "C",
        "DOCKER": "C",
        "LOG": "C",
        "JENKINS": "C",
    }
    system_room_map = {
        "APP": ["sza", "szb", "hfa"],
        "ACT": ["sza", "szb", "hfa"],
        "PAY": ["bja", "bjb", "hfa"],
        "SHOP": ["sza", "szb", "hfa"],
        "EMS": ["bja", "bjb", "hfa"],
        "DCOS": ["sza", "szb", "hfa"],
        "MAIL": ["sza", "szb", "hfa"],
        "MSG": ["bja", "bjb", "hfa"],
        "CHAT": ["bja", "bjb", "hfa"],
        "CALLCENTER": ["sza", "szb", "hfa"],
        "OA": ["sza", "szb", "hfa"],
        "CMDB": ["bja", "bjb", "hfa"],
        "NATO": ["sza", "szb"],
        "CHGM": ["bja", "bjb"],
        "YUM": ["bja", "bjb", "hfa"],
        "HABOR": ["bja", "bjb", "hfa"],
        "DOCKER": ["bja", "bjb", "hfa"],
        "LOG": ["bja", "bjb", "hfa"],
        "JENKINS": ["bja", "bjb", "hfa"],
    }
    return systems_services_map, system_level_map, system_room_map


def configure_network_mappings():
    # 网络区域与 IP 段映射
    zone_a_segment_map = {
        "AG": "10.1", "SCYW": "10.2", "KFCS": "10.3", "BD": "10.4",
        "BG": "10.5", "WEB": "10.6", "HXYW": "10.7", "OA": "10.8",
    }
    # 机房与 C 段映射
    room_c_segment_map = {"sza": 1, "szb": 2, "hfa": 3, "bja": 4, "bjb": 5}
    return zone_a_segment_map, room_c_segment_map


# 随机生成 IP 地址
def generate_ip(zone_a_segment_map, room_c_segment_map, zone, room):
    a_segment = zone_a_segment_map[zone]
    c_segment = room_c_segment_map[room]
    return f"{a_segment}.{c_segment}.{random.randint(1, 254)}"


# hfa 机房随机降配
def get_hfaflavor(room, shared_flavor, flavors):
    if room == "hfa":
        # 为 hfa 机房随机降配
        index = flavors.index(shared_flavor)
        # 返回当前规格及其之前的一个随机规格
        return random.choice(flavors[:index + 1])  # 包括当前规格及之前的规格
    return shared_flavor  # 非 hfa 规格无需降配


# 分配机器名称
def generate_machine_name(room, system, service, counter):
    return f"{room}-{system.lower()}-{service}-kzx-{counter:04d}"


# 为其他服务生成节点数，2个节点为大概率
def generate_other_service_nodes():
    return random.choices([2, 3, 4, 5, 6], weights=[85, 5, 5, 3, 2], k=1)[0]


# 从区间中随机生成字段值
def generate_random_metric(intervals, weights):
    interval = random.choices(intervals, weights=weights, k=1)[0]  # 随机选择一个区间
    return random.randint(interval[0], interval[1])  # 从区间中随机取值


# 根据输入数字返回它属于 cpu_20_intervals 的第几个区间
def find_interval_index(value, intervals):
    for index, (start, end) in enumerate(intervals):
        if start <= value <= end:
            return index + 1  # 区间从 1 开始计数
    return "输入值不在任何区间内"  # 提示超出范围


# 模拟生成性能指标数据
def generate_cpu_memory_matrix(utilization_intervals, use_type):
    cpu_utilization = generate_random_metric(utilization_intervals["CPU"]["intervals"],
                                             utilization_intervals["CPU"]["weights"][use_type])
    memory_utilization = generate_random_metric(utilization_intervals["MEMORY"]["intervals"],
                                                utilization_intervals["MEMORY"]["weights"][use_type])

    # 根据 CPU 利用率生成 CPU 时间占比
    if cpu_utilization < 20:
        cpu_20 = 0
        cpu_30 = 0
    elif cpu_utilization < 30:
        cpu_20 = generate_random_metric(utilization_intervals["CPU20"]["intervals"][1:],
                                        utilization_intervals["CPU20"]["weights"][use_type][1:])
        cpu_30 = 0
    else:
        cpu_20 = generate_random_metric(utilization_intervals["CPU20"]["intervals"][1:],
                                        utilization_intervals["CPU20"]["weights"][use_type][1:])
        if find_interval_index(cpu_20, utilization_intervals["CPU20"]["intervals"]) <= 2:
            cpu_30 = max(0, int(cpu_20 / 2))
        else:
            cpu_30 = generate_random_metric(
                utilization_intervals["CPU30"]["intervals"][
                1:max(2, find_interval_index(cpu_20, utilization_intervals["CPU20"]["intervals"]) - 1)],
                utilization_intervals["CPU30"]["weights"][use_type][
                1:max(2, find_interval_index(cpu_20,
                                             utilization_intervals["CPU20"]["intervals"]) - 1)])  # 确保 cpu_30 <= cpu_20
    return cpu_utilization, memory_utilization, cpu_20, cpu_30


def get_utilization_values(data, system, service, zone, room):
    """
    获取指定 system、service、zone 和 room 下所有机器的利用率 value 枚举值。
    """
    if system in data and service in data[system] and zone in data[system][service] \
            and room in data[system][service][zone]:
        room_data = data[system][service][zone][room]
        return set(room_data.values())  # 使用 set 去重
    else:
        return set()  # 返回空集表示没有找到


# 生成服务器数据
def generate_server_data(
        systems_services_map, system_level_map, system_room_map,
        zone_a_segment_map, room_c_segment_map, flavors, architectures,
        service_node_count, utilization_intervals, service_deploy_type):
    # 随机生成服务器数据
    server_data = []
    service_counters = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    chip_architecture_map = defaultdict(dict)
    matrix_map = {}
    flavor_map = {}

    # 为每个系统和服务，确保网络区域唯一
    system_service_zone_map = defaultdict(lambda: defaultdict(str))

    # 定义系统-服务-网络区域-机房下各机器热、温、冷特性
    system_service_zone_room_server_utilization_map = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(str)))))

    for system, services in systems_services_map.items():
        level = system_level_map[system]  # 系统级别
        rooms = system_room_map[system]  # 系统对应机房
        deploy_type = random.choice(list(service_deploy_type.keys()))  # 部署方案

        for service in services:
            # 确定网络区域，确保相同系统、服务在同一机房下的网络区域一致
            if system_service_zone_map[system][service] == "":
                # 如果该服务的网络区域还未分配，随机选择一个网络区域
                if system in ["PAY", "SHOP", "EMS", "MAIL", "CHAT", "MSG"]:
                    zone = random.choice(["WEB", "SCYW", "AG"])
                else:
                    zone = random.choice(["AG", "SCYW", "HXYW", "KFCS", "OA"])
                system_service_zone_map[system][service] = zone  # 保存服务的网络区域

            zone = system_service_zone_map[system][service]

            # 根据服务类型定义节点数
            if service in service_node_count:
                num_nodes = service_node_count[service]
            else:
                num_nodes = generate_other_service_nodes()  # 其他服务的节点数

            for room in rooms:
                # 选择架构
                chip_architecture_map[system].setdefault(room, {})  # 初始化 room 为一个空字典（如果它不存在）
                if service not in chip_architecture_map[system][room]:
                    chip_architecture_map[system][room][service] = random.choice(architectures)
                architecture = chip_architecture_map[system][room][service]

                # 确保规格一致，除 hfa 外
                if (system, service) not in flavor_map:
                    shared_flavor = random.choice(flavors)  # 随机选择规格
                    flavor_map[(system, service)] = shared_flavor  # 保存该系统、服务的规格
                else:
                    shared_flavor = flavor_map[(system, service)]
                # 为 hfa 机房分配较低规格
                flavor = get_hfaflavor(room, shared_flavor, flavors)

                if room == "hfa" and service not in service_node_count:
                    num_nodes = random.randint(2, num_nodes)

                for _ in range(num_nodes):
                    ip = generate_ip(zone_a_segment_map, room_c_segment_map, zone, room)
                    counter = service_counters[room][system][service]
                    machine_name = generate_machine_name(room, system, service, counter + 1)
                    service_counters[room][system][service] += 1

                    # 服务器运行态资源使用率
                    if room == "hfa":  # hfa全部为冷机器
                        system_service_zone_room_server_utilization_map[system][service][zone][room][
                            machine_name] = 'low'
                    else:
                        # 第一台机器为热机器
                        if len(system_service_zone_room_server_utilization_map[system][service][zone]) == 0:
                            system_service_zone_room_server_utilization_map[system][service][zone][room][
                                machine_name] = 'high'
                        elif "DDAA" in deploy_type:  # 机房级双活, 按照机房区分high、medium
                            if len(system_service_zone_room_server_utilization_map[system][service][zone][room]) == 0:
                                system_service_zone_room_server_utilization_map[system][service][zone][room][
                                    machine_name] = 'medium'
                            else:
                                system_service_zone_room_server_utilization_map[system][service][zone][room][
                                    machine_name] = get_utilization_values(
                                    system_service_zone_room_server_utilization_map, system, service, zone, room)[0]
                        elif "DDAS" in deploy_type:  # 机房级主备, 按照机房区分high、low
                            if len(system_service_zone_room_server_utilization_map[system][service][zone][room]) == 0:
                                system_service_zone_room_server_utilization_map[system][service][zone][room][
                                    machine_name] = 'low'
                            else:
                                system_service_zone_room_server_utilization_map[system][service][zone][room][
                                    machine_name] = get_utilization_values(
                                    system_service_zone_room_server_utilization_map, system, service, zone, room)[0]
                        elif "CAA" in deploy_type:  # 集群级多活, 按照集群区分high、medium、low
                            if 'high' in get_utilization_values(system_service_zone_room_server_utilization_map, system,
                                                                service, zone, room):
                                system_service_zone_room_server_utilization_map[system][service][zone][room][
                                    machine_name] = 'medium'
                            else:
                                system_service_zone_room_server_utilization_map[system][service][zone][room][
                                    machine_name] = 'low'
                        elif "CAS" in deploy_type:  # 集群级主备, 按照集群区分high、low
                            system_service_zone_room_server_utilization_map[system][service][zone][room][
                                machine_name] = 'low'

                    print(
                        "system   service zone    room    machine_name    type:\n" + system + "   " + service + "   " + zone + "   " + room + "   " + machine_name +
                        system_service_zone_room_server_utilization_map[system][service][zone][room][machine_name])
                    cpu_matrix, memory_matrix, cpu_20matrix, cpu_30matrix = generate_cpu_memory_matrix(
                        utilization_intervals,
                        system_service_zone_room_server_utilization_map[system][service][zone][room][machine_name])

                    server_data.append({
                        "label": random.choice(
                            ["0-平", "1-升CPU", "2-升内存", "3-升CPU内存", "4-降CPU", "5-降内存", "6-降CPU内存"]),
                        # "冷热情况": system_service_zone_room_server_utilization_map[system][service][zone][room][machine_name],
                        "机器名": machine_name,
                        "业务IP": ip,
                        "归属系统": system,
                        "归属微服务": service,
                        "系统级别": level,
                        "进程": service,
                        "规格": flavor,
                        "机房": room,
                        "网络区域": zone,
                        "芯片架构": architecture,
                        "最高CPU利用率": f"{cpu_matrix / 100: .2f}",
                        "最高内存利用率": f"{memory_matrix / 100: .2f}",
                        "CPU利用率(>20%)时间占比": f"{cpu_20matrix / 100: .2f}",
                        "CPU利用率(>30%)时间占比": f"{cpu_30matrix / 100: .2f}",
                        "创建/变更时间": f"2020-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                    })

    return server_data


# 服务器日常变更记录
def maintain_server(server_data):
    return server_data


def init_server():
    systems_services_map, system_level_map, system_room_map = configure_mappings()
    zone_a_segment_map, room_c_segment_map = configure_network_mappings()

    # 规格列表更新
    flavors = ["2C4G", "2C8G", "4C8G", "4C16G", "8C16G", "8C32G", "16C32G", "16C64G", "32C64G"]
    architectures = ["x86", "ARM", "C86"]

    # 定义每个服务的节点数及其分布概率
    service_node_count = {
        "Redis": 6,  # Redis 每个网络区域 6 个节点
        "ES": 3,  # ES 每个网络区域 3 个节点
        "Kafka": 3,  # Kafka 每个网络区域 3 个节点
    }

    # 定义服务部署方案
    service_deploy_type = {
        "DCAA": "机房级多活",
        "DCAS": "机房级主备",
        "CAA": "集群级多活",
        "CAS": "集群级主备",
    }

    # 定义性能数据区间
    utilization_intervals = {
        "CPU": {
            "intervals": [(10, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 79), (80, 89), (90, 99),
                          (100, 100)],
            "weights": {
                "high": [1, 2, 3, 3, 3, 10, 20, 30, 18, 10],
                "medium": [3, 5, 10, 15, 17, 17, 15, 10, 5, 3],
                "low": [25, 22, 15, 11, 7, 6, 5, 4, 3, 2]
            }
        },
        "MEMORY": {
            "intervals": [(10, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 79), (80, 89), (90, 99),
                          (100, 100)],
            "weights": {
                "high": [1, 2, 3, 3, 3, 10, 20, 30, 18, 10],
                "medium": [3, 5, 10, 15, 17, 17, 15, 10, 5, 3],
                "low": [25, 22, 15, 11, 7, 6, 5, 4, 3, 2]
            }
        },
        "CPU20": {
            "intervals": [(0, 0), (1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60), (61, 70), (71, 80),
                          (81, 90), (91, 100)],
            "weights": {
                "high": [0, 1, 1, 2, 2, 10, 20, 40, 18, 3, 3],
                "medium": [0, 10, 15, 15, 17, 17, 10, 5, 5, 3, 3],
                "low": [10, 25, 25, 15, 9, 6, 4, 3, 2, 1, 0]
            }
        },
        "CPU30": {
            "intervals": [(0, 0), (1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60), (61, 70), (71, 80),
                          (81, 90), (91, 100)],
            "weights": {
                "high": [1, 2, 3, 4, 4, 12, 20, 30, 18, 3, 3],
                "medium": [10, 10, 13, 13, 15, 15, 8, 5, 5, 3, 3],
                "low": [15, 25, 22, 15, 9, 6, 3, 2, 2, 1, 0]
            }
        }
    }

    # 定义每个服务的节点数及其分布概率
    service_node_count = {
        "Redis": 6,  # Redis 每个网络区域 6 个节点
        "ES": 3,  # ES 每个网络区域 3 个节点
        "Kafka": 3,  # Kafka 每个网络区域 3 个节点
    }

    project_dir = "D:/Dream Future Project/workspace/pytorchDemo/"

    server_data = generate_server_data(
        systems_services_map, system_level_map, system_room_map,
        zone_a_segment_map, room_c_segment_map, flavors, architectures,
        service_node_count, utilization_intervals, service_deploy_type)

    server_data = maintain_server(server_data)

    # 转换为 DataFrame
    df = pd.DataFrame(server_data)

    # 排序：系统、服务、网络区域、机房
    df = df.sort_values(by=["归属系统", "归属微服务", "网络区域", "机房"])

    # 保存为 UTF-8 编码，并解决中文乱码问题
    df.to_csv(project_dir+"dataset/SERVER/server_list.csv", index=False, encoding="utf-8-sig")
    print(df.head())  # 打印部分记录验证


if __name__ == "__main__":
    init_server()
