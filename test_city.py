import itertools
import random
import sys
import warnings
from multiprocessing import Pool

import clustering

warnings.filterwarnings("ignore")
import networkx as nx
import pandas as pd
from tqdm import tqdm
# import matplotlib.pyplot as plt
import graph_tests
import graph_generator

import osmnx as ox
from common import GraphLayer


def get_graph(city_id: str = 'R2555133') -> nx.Graph:
    gdf = ox.geocode_to_gdf(city_id, by_osmid=True)
    polygon_boundary = gdf.unary_union
    graph = ox.graph_from_polygon(polygon_boundary,
                                  network_type='drive',
                                  simplify=True)
    G = nx.Graph(graph)
    H = nx.Graph()
    # Добавляем рёбра в новый граф, копируя только веса
    for u, d in G.nodes(data=True):
        H.add_node(u, x=d['x'], y=d['y'])
    for u, v, d in G.edges(data=True):
        H.add_edge(u, v, length=d['length'])
    del city_id, gdf, polygon_boundary, graph, G
    return H


def percent(l, p=0.1, max_points=10):
    res = int(len(l) * p) if len(l) * p >= 1 else 1
    if res > max_points:
        res = 10
    return res


def find_points_for_experiment(l: GraphLayer, min_length=5.0, p=0.1, max_points=2):
    # словарь хранит в себе {id_кластер: {other_кластер: расстояние до него}}
    d = dict(nx.all_pairs_dijkstra_path_length(l.centroids_graph, weight='length'))
    df = pd.DataFrame(d)
    df = df.reindex(sorted(df.columns), axis=1)
    df = df.sort_index()
    graph_path_length = df.values
    res = []
    for com_1 in range(len(graph_path_length)):
        for com_2 in range(com_1 + 1, len(graph_path_length)):
            if graph_path_length[com_1][com_2] >= min_length:
                list_1 = random.choices(list(l.communities[com_1]),
                                        k=percent(l.communities[com_1], p, max_points))
                list_2 = random.choices(list(l.communities[com_2]),
                                        k=percent(l.communities[com_2], p, max_points))
                all_lists = itertools.product(list_1, list_2)

                res.extend(list(all_lists))
    try:
        part = random.choices(res, k=max_points)
    except ValueError:
        part = res  # Если k больше размера популяции или отрицательно, используем весь список
    print(len(part))
    return part


def calculate(data):
    city = data[0]
    city_id = data[1]
    points_number = data[2]

    G = get_graph(city_id)
    Q = G.copy()
    for u in Q.nodes:
        if u in Q[u]:
            Q.remove_edge(u, u)

    # l,_,_,_ = graph_generator.generate_layer(Q, 1)
    # points = find_points_for_experiment(l, min_length=nx.radius(Q, weight='length') * 0.5, max_points=points_number)
    points = [graph_generator.get_node_for_initial_graph_v2(G) for _ in
              range(points_number)]
    #    graph_tests.test_graph(Q,
    #                          f'{city}',
    #                          city,
    #                          points=points, logs=True)
    usual_results = graph_tests.get_usual_result(Q, points, alg='dijkstra')
    methods = [clustering.Method.LOUVAIN, clustering.Method.LOUVAIN_K_MEANS, clustering.Method.BISECTING_K_MEAN,
               clustering.Method.K_MEAN, clustering.Method.GREEDY_MODULARITY]
    for m in tqdm([clustering.Method.LOUVAIN], position=2):
        graph_tests.test_graph(Q,
                               name=f'{city}',
                               city_id=city_id,
                               points=points,
                               clustering=m, usual_results=usual_results)


if __name__ == '__main__':
    total = 1
    points_number = 1000
    if len(sys.argv) == 2:
        total = int(sys.argv[1])
    if len(sys.argv) == 3:
        total = int(sys.argv[1])
        points_number = int(sys.argv[2])
    # print('THREADS:', total)
    # print('POINTS:', points_number)

    cities = {
        # 'ASHA': 'R13470549',
        # 'KRG': 'R4676636',
        # 'EKB': 'R6564910',
        # 'BARCELONA': 'R347950',
        # 'PARIS': 'R71525',
        # 'Prague': 'R435514',
        'MSK': 'R2555133',
        # 'SBP': 'R337422',
        # 'SINGAPORE': 'R17140517',
        # 'BERLIN': 'R62422',
        # 'ROME': 'R41485',
        # 'LA': 'R207359',
        # 'DUBAI': 'R4479752',
        # 'RIO': 'R2697338',
        # 'DELHI': 'R1942586',
        # 'KAIR': 'R5466227'
    }

    # cities = {'R1312868': 'R1312868', 'R1418311': 'R1418311', 'R3377820': 'R3377820', 'R963764': 'R963764',
    #           'R175481': 'R175481', 'R2682926': 'R2682926', 'R963794': 'R963794', 'R2689476': 'R2689476',
    #           'R253824': 'R253824', 'R3377873': 'R3377873', 'R3377835': 'R3377835', 'R3936114': 'R3936114',
    #           'R3441283': 'R3441283', 'R1437127': 'R1437127', 'R3939930': 'R3939930', 'R2547126': 'R2547126',
    #           'R3031682': 'R3031682', 'R2679920': 'R2679920', 'R2296668': 'R2296668', 'R117709': 'R117709',
    #           'R2682914': 'R2682914', 'R3030295': 'R3030295', 'R2682922': 'R2682922', 'R1748539': 'R1748539',
    #           'R2049848': 'R2049848', 'R174387': 'R174387', 'R1670935': 'R1670935', 'R1153174': 'R1153174',
    #           'R1599637': 'R1599637', 'R182887': 'R182887', 'R2356000': 'R2356000', 'R3488359': 'R3488359',
    #           'R119990': 'R119990', 'R2728438': 'R2728438', 'R3437391': 'R3437391', 'R124821': 'R124821',
    #           'R2069683': 'R2069683', 'R2105705': 'R2105705', 'R3390600': 'R3390600', 'R2623018': 'R2623018',
    #           'R347950': 'R347950', 'R2220322': 'R2220322', 'R119569': 'R119569', 'R183421': 'R183421',
    #           'R239887': 'R239887', 'R1933719': 'R1933719', 'R1437129': 'R1437129', 'R2682932': 'R2682932',
    #           'R1390623': 'R1390623', 'R3437172': 'R3437172', 'R3511787': 'R3511787', 'R111825': 'R111825',
    #           'R2689429': 'R2689429', 'R132145': 'R132145', 'R3388835': 'R3388835', 'R1759474': 'R1759474',
    #           'R2999176': 'R2999176', 'R1749244': 'R1749244', 'R2679906': 'R2679906', 'R2679914': 'R2679914',
    #           'R1118893': 'R1118893', 'R2679912': 'R2679912', 'R1382820': 'R1382820', 'R4145881': 'R4145881',
    #           'R132206': 'R132206', 'R2408838': 'R2408838', 'R119959': 'R119959', 'R184985': 'R184985',
    #           'R194124': 'R194124', 'R2383939': 'R2383939', 'R2689426': 'R2689426', 'R1839563': 'R1839563',
    #           'R2682908': 'R2682908', 'R3160177': 'R3160177', 'R125650': 'R125650', 'R2521434': 'R2521434',
    #           'R2628521': 'R2628521', 'R125785': 'R125785', 'R4198908': 'R4198908', 'R1735835': 'R1735835',
    #           'R2758781': 'R2758781', 'R174979': 'R174979', 'R125411': 'R125411', 'R2318282': 'R2318282',
    #           'R3348896': 'R3348896', 'R180114': 'R180114', 'R62407': 'R62407', 'R3154746': 'R3154746',
    #           'R3158297': 'R3158297', 'R2679930': 'R2679930', 'R2410555': 'R2410555', 'R2688911': 'R2688911',
    #           'R2320570': 'R2320570', 'R1768502': 'R1768502', 'R3649003': 'R3649003', 'R174916': 'R174916',
    #           'R3163676': 'R3163676', 'R2682890': 'R2682890', 'R1768272': 'R1768272', 'R2682925': 'R2682925',
    #           'R1109531': 'R1109531', 'R1768312': 'R1768312', 'R3309365': 'R3309365', 'R1382494': 'R1382494',
    #           'R181886': 'R181886', 'R62430': 'R62430', 'R3308306': 'R3308306', 'R1865315': 'R1865315',
    #           'R2689444': 'R2689444', 'R2164745': 'R2164745', 'R3554015': 'R3554015', 'R2175059': 'R2175059',
    #           'R2538203': 'R2538203', 'R1792913': 'R1792913', 'R2679904': 'R2679904', 'R2679938': 'R2679938',
    #           'R1517394': 'R1517394', 'R3277038': 'R3277038', 'R2878463': 'R2878463', 'R2387995': 'R2387995',
    #           'R3308039': 'R3308039', 'R198772': 'R198772', 'R1282148': 'R1282148', 'R3476238': 'R3476238',
    #           'R2049867': 'R2049867', 'R3305920': 'R3305920', 'R2682910': 'R2682910', 'R175466': 'R175466',
    #           'R1752948': 'R1752948', 'R2350024': 'R2350024', 'R119373': 'R119373', 'R2396450': 'R2396450',
    #           'R1801797': 'R1801797', 'R2416081': 'R2416081', 'R3396088': 'R3396088', 'R1760124': 'R1760124',
    #           'R194134': 'R194134', 'R134765': 'R134765', 'R963787': 'R963787', 'R111968': 'R111968',
    #           'R1430614': 'R1430614', 'R2344391': 'R2344391', 'R1128379': 'R1128379', 'R389790': 'R389790',
    #           'R1865772': 'R1865772', 'R4188047': 'R4188047', 'R2131434': 'R2131434', 'R2682889': 'R2682889',
    #           'R62559': 'R62559', 'R1853866': 'R1853866', 'R2190496': 'R2190496', 'R1782722': 'R1782722',
    #           'R1759477': 'R1759477', 'R1838336': 'R1838336', 'R185048': 'R185048', 'R1761743': 'R1761743',
    #           'R2712310': 'R2712310', 'R3306086': 'R3306086', 'R2875222': 'R2875222', 'R175031': 'R175031',
    #           'R2682897': 'R2682897', 'R1430616': 'R1430616', 'R3816063': 'R3816063', 'R3554346': 'R3554346',
    #           'R2682931': 'R2682931', 'R1863552': 'R1863552', 'R119734': 'R119734', 'R3058686': 'R3058686',
    #           'R1477110': 'R1477110', 'R2381773': 'R2381773', 'R131885': 'R131885', 'R1645367': 'R1645367',
    #           'R3653961': 'R3653961', 'R1768273': 'R1768273', 'R112143': 'R112143', 'R3171497': 'R3171497',
    #           'R930950': 'R930950', 'R114496': 'R114496', 'R1902682': 'R1902682', 'R1865757': 'R1865757',
    #           'R2409550': 'R2409550', 'R4196833': 'R4196833', 'R1674442': 'R1674442', 'R3520594': 'R3520594',
    #           'R119571': 'R119571', 'R1488071': 'R1488071', 'R2738421': 'R2738421', 'R2682892': 'R2682892',
    #           'R3116010': 'R3116010', 'R207359': 'R207359', 'R175487': 'R175487', 'R3540206': 'R3540206',
    #           'R3940146': 'R3940146', 'R1281564': 'R1281564', 'R3864704': 'R3864704', 'R2679948': 'R2679948',
    #           'R1202659': 'R1202659', 'R3694013': 'R3694013', 'R3406249': 'R3406249', 'R3337257': 'R3337257',
    #           'R2679921': 'R2679921', 'R1574611': 'R1574611', 'R174482': 'R174482', 'R184748': 'R184748',
    #           'R119192': 'R119192', 'R963789': 'R963789', 'R3437242': 'R3437242', 'R1430508': 'R1430508',
    #           'R3536487': 'R3536487', 'R2614178': 'R2614178', 'R136612': 'R136612', 'R3396084': 'R3396084',
    #           'R1768315': 'R1768315', 'R3554304': 'R3554304', 'R3442814': 'R3442814', 'R2866010': 'R2866010',
    #           'R3610503': 'R3610503', 'R3936275': 'R3936275', 'R3986379': 'R3986379', 'R1931997': 'R1931997',
    #           'R2682906': 'R2682906', 'R3440980': 'R3440980', 'R2689440': 'R2689440', 'R345456': 'R345456',
    #           'R1252581': 'R1252581', 'R4230155': 'R4230155', 'R1165478': 'R1165478', 'R2224013': 'R2224013',
    #           'R3862199': 'R3862199', 'R2692232': 'R2692232', 'R179034': 'R179034', 'R79911': 'R79911',
    #           'R3368701': 'R3368701', 'R181889': 'R181889', 'R119817': 'R119817', 'R174885': 'R174885',
    #           'R2679899': 'R2679899', 'R2682900': 'R2682900', 'R174472': 'R174472', 'R188481': 'R188481',
    #           'R2062152': 'R2062152', 'R119970': 'R119970', 'R3531585': 'R3531585', 'R2689422': 'R2689422',
    #           'R2679915': 'R2679915', 'R2689430': 'R2689430', 'R2228485': 'R2228485', 'R2407500': 'R2407500',
    #           'R1768314': 'R1768314', 'R130921': 'R130921', 'R2900261': 'R2900261', 'R2408844': 'R2408844',
    #           'R3363950': 'R3363950', 'R2910703': 'R2910703', 'R2122758': 'R2122758', 'R206673': 'R206673',
    #           'R125756': 'R125756', 'R3134925': 'R3134925', 'R1768336': 'R1768336', 'R2679943': 'R2679943',
    #           'R2529624': 'R2529624', 'R366544': 'R366544', 'R188022': 'R188022', 'R2506125': 'R2506125',
    #           'R194120': 'R194120', 'R119643': 'R119643', 'R361818': 'R361818', 'R1382923': 'R1382923',
    #           'R2682898': 'R2682898', 'R2682913': 'R2682913', 'R2679896': 'R2679896', 'R175050': 'R175050',
    #           'R4062443': 'R4062443', 'R1181618': 'R1181618', 'R2682923': 'R2682923', 'R1181621': 'R1181621',
    #           'R4185743': 'R4185743', 'R174494': 'R174494', 'R175464': 'R175464', 'R1382460': 'R1382460',
    #           'R1768274': 'R1768274', 'R2911978': 'R2911978', 'R2414222': 'R2414222', 'R2689481': 'R2689481',
    #           'R182777': 'R182777', 'R2679949': 'R2679949', 'R3396612': 'R3396612', 'R2679917': 'R2679917',
    #           'R963797': 'R963797', 'R963792': 'R963792', 'R1761718': 'R1761718', 'R2799215': 'R2799215',
    #           'R3940406': 'R3940406', 'R3575300': 'R3575300', 'R174933': 'R174933', 'R3511468': 'R3511468',
    #           'R125796': 'R125796', 'R2032280': 'R2032280', 'R4116163': 'R4116163', 'R400507': 'R400507',
    #           'R1929810': 'R1929810', 'R2049858': 'R2049858', 'R119370': 'R119370', 'R1491893': 'R1491893',
    #           'R1382458': 'R1382458', 'R1933746': 'R1933746', 'R1544955': 'R1544955', 'R3866463': 'R3866463',
    #           'R2679924': 'R2679924', 'R2679925': 'R2679925', 'R1438336': 'R1438336', 'R2689445': 'R2689445',
    #           'R2383991': 'R2383991', 'R113008': 'R113008', 'R2062154': 'R2062154', 'R1749243': 'R1749243',
    #           'R134897': 'R134897', 'R3642177': 'R3642177', 'R174883': 'R174883', 'R3313703': 'R3313703',
    #           'R113954': 'R113954', 'R110691': 'R110691', 'R1861888': 'R1861888', 'R2679907': 'R2679907',
    #           'R3374767': 'R3374767', 'R2679923': 'R2679923', 'R119565': 'R119565', 'R4155387': 'R4155387',
    #           'R4089182': 'R4089182', 'R174399': 'R174399', 'R176069': 'R176069', 'R2689438': 'R2689438',
    #           'R2682901': 'R2682901', 'R62644': 'R62644', 'R3835109': 'R3835109', 'R3338286': 'R3338286',
    #           'R2401094': 'R2401094', 'R174952': 'R174952', 'R1926754': 'R1926754', 'R3562367': 'R3562367',
    #           'R1701311': 'R1701311', 'R191645': 'R191645', 'R1555410': 'R1555410', 'R299354': 'R299354',
    #           'R2679901': 'R2679901', 'R2865045': 'R2865045', 'R206637': 'R206637', 'R2479316': 'R2479316',
    #           'R2679937': 'R2679937', 'R2283083': 'R2283083', 'R136670': 'R136670', 'R175549': 'R175549',
    #           'R2682920': 'R2682920', 'R1360937': 'R1360937', 'R963784': 'R963784', 'R1933055': 'R1933055',
    #           'R1382493': 'R1382493', 'R2407513': 'R2407513', 'R3807613': 'R3807613', 'R3374737': 'R3374737',
    #           'R2530671': 'R2530671', 'R1988678': 'R1988678', 'R1758878': 'R1758878', 'R1429283': 'R1429283',
    #           'R2407259': 'R2407259', 'R182882': 'R182882', 'R1759506': 'R1759506', 'R163244': 'R163244',
    #           'R3940067': 'R3940067', 'R3678529': 'R3678529', 'R174869': 'R174869', 'R398021': 'R398021',
    #           'R2069556': 'R2069556', 'R1840167': 'R1840167', 'R324211': 'R324211', 'R345039': 'R345039',
    #           'R3511823': 'R3511823', 'R125791': 'R125791', 'R2689441': 'R2689441', 'R2682907': 'R2682907',
    #           'R1758947': 'R1758947', 'R1629268': 'R1629268', 'R1768399': 'R1768399', 'R963777': 'R963777',
    #           'R3555552': 'R3555552', 'R1653741': 'R1653741', 'R119694': 'R119694', 'R1413957': 'R1413957',
    #           'R1954127': 'R1954127', 'R2682927': 'R2682927', 'R2555133': 'R2555133', 'R2222157': 'R2222157',
    #           'R3578080': 'R3578080', 'R2407358': 'R2407358', 'R112912': 'R112912', 'R1248301': 'R1248301',
    #           'R2171253': 'R2171253', 'R2910227': 'R2910227', 'R2689443': 'R2689443', 'R1758897': 'R1758897',
    #           'R2730507': 'R2730507', 'R963782': 'R963782', 'R131703': 'R131703', 'R2498868': 'R2498868',
    #           'R119639': 'R119639', 'R2048253': 'R2048253', 'R3616563': 'R3616563', 'R2408836': 'R2408836',
    #           'R1181614': 'R1181614', 'R2793022': 'R2793022', 'R1327509': 'R1327509', 'R1306984': 'R1306984',
    #           'R1761717': 'R1761717', 'R3099384': 'R3099384', 'R2401506': 'R2401506', 'R2647900': 'R2647900',
    #           'R2387988': 'R2387988', 'R127408': 'R127408', 'R2408837': 'R2408837', 'R174495': 'R174495',
    #           'R2465058': 'R2465058', 'R3678531': 'R3678531', 'R1860880': 'R1860880', 'R206666': 'R206666',
    #           'R2682928': 'R2682928', 'R2689433': 'R2689433', 'R324212': 'R324212', 'R137238': 'R137238',
    #           'R169588': 'R169588', 'R1746879': 'R1746879', 'R2689442': 'R2689442', 'R2221709': 'R2221709',
    #           'R119521': 'R119521', 'R2383150': 'R2383150', 'R1489187': 'R1489187', 'R2131460': 'R2131460',
    #           'R2586928': 'R2586928', 'R1863553': 'R1863553', 'R2068203': 'R2068203', 'R1758936': 'R1758936',
    #           'R118879': 'R118879', 'R1381350': 'R1381350', 'R1645570': 'R1645570', 'R119567': 'R119567',
    #           'R1768402': 'R1768402', 'R115275': 'R115275', 'R3551849': 'R3551849', 'R2131479': 'R2131479',
    #           'R1942017': 'R1942017', 'R183453': 'R183453', 'R2295550': 'R2295550', 'R3308780': 'R3308780',
    #           'R346782': 'R346782', 'R3437417': 'R3437417', 'R2997809': 'R2997809', 'R2689421': 'R2689421',
    #           'R1517207': 'R1517207', 'R3565917': 'R3565917', 'R421866': 'R421866', 'R184964': 'R184964',
    #           'R125398': 'R125398', 'R3864703': 'R3864703', 'R167857': 'R167857', 'R1549169': 'R1549169',
    #           'R2773264': 'R2773264', 'R206672': 'R206672', 'R2679911': 'R2679911', 'R3306997': 'R3306997',
    #           'R169681': 'R169681', 'R178128': 'R178128', 'R3087156': 'R3087156', 'R1380013': 'R1380013',
    #           'R119353': 'R119353', 'R2679918': 'R2679918', 'R2689424': 'R2689424', 'R1515474': 'R1515474',
    #           'R333748': 'R333748', 'R1482451': 'R1482451', 'R4196796': 'R4196796', 'R1775054': 'R1775054',
    #           'R1285772': 'R1285772', 'R1758888': 'R1758888', 'R2409549': 'R2409549', 'R2689425': 'R2689425',
    #           'R2679934': 'R2679934', 'R1749248': 'R1749248', 'R175905': 'R175905', 'R3377881': 'R3377881',
    #           'R2679929': 'R2679929', 'R1361875': 'R1361875', 'R1934961': 'R1934961', 'R2762261': 'R2762261',
    #           'R1748490': 'R1748490', 'R127409': 'R127409', 'R2380421': 'R2380421', 'R2682893': 'R2682893',
    #           'R3992676': 'R3992676', 'R3106939': 'R3106939', 'R2679900': 'R2679900', 'R2682891': 'R2682891',
    #           'R2689427': 'R2689427', 'R3864686': 'R3864686', 'R1931999': 'R1931999', 'R963769': 'R963769',
    #           'R1768313': 'R1768313', 'R3030976': 'R3030976', 'R2679910': 'R2679910', 'R113329': 'R113329',
    #           'R2679905': 'R2679905', 'R3536492': 'R3536492', 'R112937': 'R112937', 'R1330715': 'R1330715',
    #           'R3311487': 'R3311487', 'R2682899': 'R2682899', 'R3404703': 'R3404703', 'R3353648': 'R3353648',
    #           'R1760040': 'R1760040', 'R1543056': 'R1543056', 'R119557': 'R119557', 'R115339': 'R115339',
    #           'R2168517': 'R2168517', 'R2682917': 'R2682917', 'R2062153': 'R2062153', 'R123419': 'R123419',
    #           'R2682912': 'R2682912', 'R176222': 'R176222', 'R2027318': 'R2027318', 'R2407406': 'R2407406',
    #           'R62422': 'R62422', 'R1316250': 'R1316250', 'R2682921': 'R2682921', 'R1768335': 'R1768335',
    #           'R1118894': 'R1118894', 'R3407206': 'R3407206', 'R2307824': 'R2307824', 'R2679913': 'R2679913',
    #           'R181320': 'R181320', 'R3308973': 'R3308973', 'R1769181': 'R1769181', 'R2679928': 'R2679928',
    #           'R122604': 'R122604', 'R1742393': 'R1742393', 'R181304': 'R181304', 'R3585163': 'R3585163',
    #           'R3437213': 'R3437213', 'R2689434': 'R2689434', 'R2635526': 'R2635526', 'R1343264': 'R1343264',
    #           'R3561691': 'R3561691', 'R2679935': 'R2679935', 'R3912206': 'R3912206', 'R176131': 'R176131',
    #           'R1933745': 'R1933745', 'R119867': 'R119867', 'R1704857': 'R1704857', 'R117441': 'R117441',
    #           'R1842139': 'R1842139', 'R3377895': 'R3377895', 'R1703080': 'R1703080', 'R2263374': 'R2263374',
    #           'R3629362': 'R3629362', 'R1712780': 'R1712780', 'R2632192': 'R2632192', 'R2866412': 'R2866412',
    #           'R1758858': 'R1758858', 'R1879609': 'R1879609', 'R1379449': 'R1379449', 'R1550512': 'R1550512',
    #           'R2122756': 'R2122756', 'R3374641': 'R3374641', 'R2682918': 'R2682918', 'R3348901': 'R3348901',
    #           'R2297418': 'R2297418', 'R1805416': 'R1805416', 'R170683': 'R170683', 'R1839150': 'R1839150',
    #           'R956596': 'R956596', 'R376977': 'R376977', 'R1181609': 'R1181609', 'R184109': 'R184109',
    #           'R2689431': 'R2689431', 'R1842140': 'R1842140', 'R182896': 'R182896', 'R119572': 'R119572',
    #           'R1667827': 'R1667827', 'R1867896': 'R1867896', 'R175053': 'R175053', 'R2866485': 'R2866485',
    #           'R62713': 'R62713', 'R1524228': 'R1524228', 'R2689482': 'R2689482', 'R1582236': 'R1582236',
    #           'R324213': 'R324213', 'R3726744': 'R3726744', 'R1440528': 'R1440528', 'R1877091': 'R1877091',
    #           'R2362670': 'R2362670', 'R1768338': 'R1768338', 'R1758891': 'R1758891', 'R206561': 'R206561',
    #           'R174454': 'R174454', 'R115305': 'R115305', 'R2414122': 'R2414122', 'R2388057': 'R2388057',
    #           'R2071804': 'R2071804', 'R119641': 'R119641', 'R134591': 'R134591', 'R2682909': 'R2682909',
    #           'R114485': 'R114485', 'R119367': 'R119367', 'R2679940': 'R2679940', 'R3308775': 'R3308775',
    #           'R175727': 'R175727', 'R3940400': 'R3940400', 'R113963': 'R113963', 'R1420221': 'R1420221',
    #           'R2689419': 'R2689419', 'R181934': 'R181934', 'R62145': 'R62145'}
    total_len = len(cities)
    l = list(cities.items())
    data = [[name, cities[name], points_number] for name in cities]
    print(total)
    with Pool(total) as p:
        result = list(tqdm(p.imap_unordered(calculate, data), total=len(data)))
