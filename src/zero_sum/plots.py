import matplotlib.pyplot as plt
import numpy as np

strategies_old = np.array([
    [
        #gamma = 0.5
        [-9.563336572173805, -8.193748914742143, -10.270220524396596, -3.0000000000000004, -7.553846153846153, -7.904142011834319, -3.0, -3.0, -3.0, -3.0],
        [-7.378487640724603, -3.3197512739857147, -7.496142688168831, -3.0000000000000004, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0], 
        [-9.314285714285715, -6.0, -11.8, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0]
    ],
    [
        #gamma = 0.55
        [-10.524329851693846, -9.121195380360348, -11.382794510864285, -3.333333333333334, -8.308123249299719, -8.767977779347031, -3.333333333333334, -3.333333333333334, -3.333333333333334, -3.333333333333334],
        [-7.853847828218031, -3.71793284163171, -7.966237505840968, -3.333333333333334, -3.333333333333333, -3.333333333333333, -3.333333333333334, -3.333333333333334, -3.333333333333334, -3.333333333333334], 
        [-10.028985507246377, -6.666666666666666, -13.11111111111111, -6.666666666666666, -6.666666666666666, -6.666666666666666, -6.666666666666666, -6.666666666666666, -6.666666666666666, -6.666666666666666]
    ],
    [
        #gamma = 0.6
        [-11.719684181565645, -10.273338406900049, -12.76628614916286, -3.75, -9.231481481481477, -9.840534979423865, -3.75, -3.75, -3.75, -3.75],
        [-8.42965572232598, -4.21204039850597, -8.528273788767603, -3.7499999999999982, -3.7499999999999982, -3.7499999999999982, -3.7499999999999982, -3.7499999999999982, -3.7499999999999982, -3.7499999999999982], 
        [-10.911764705882351, -7.499999999999999, -14.441176470588236, -7.499999999999999, -7.499999999999999, -7.499999999999999, -7.499999999999999, -7.499999999999999, -7.499999999999999, -7.499999999999999]
    ],
    [
        #gamma = 0.65
        [-13.246274902675406, -11.742746583844642, -14.533245644719841, -4.285714285714285, -10.388807069219439, -11.206747339173734, -4.285714285714285, -4.285714285714285, -4.285714285714285, -4.285714285714285], 
        [-9.145905020699661, -4.841618667247477, -9.218958552423087, -4.285714285714285, -4.285714285714285, -4.285714285714285, -4.285714285714283, -4.285714285714283, -4.285714285714283, -4.285714285714283], 
        [-12.03411513859275, -8.57142857142857, -15.616204690831555, -8.57142857142857, -8.57142857142857, -8.57142857142857, -8.57142857142857, -8.57142857142857, -8.57142857142857, -8.57142857142857]
    ],
    [
        #gamma = 0.7
        [-15.26280962470669, -13.681019910677742, -16.868493443177165, -4.999999999999998, -11.883720930232553, -13.004326663061104, -4.999999999999998, -4.999999999999998, -4.999999999999998, -4.999999999999998], 
        [-10.068193838520614, -5.671752735193775, -10.099054988853647, -4.999999999999998, -4.999999999999998, -4.999999999999998, -4.999999999999998, -4.999999999999998, -4.999999999999998, -4.999999999999998], 
        [-13.515151515151512, -9.999999999999996, -17.15151515151515, -9.999999999999996, -9.999999999999996, -9.999999999999996, -9.999999999999996, -9.999999999999996, -9.999999999999996, -9.999999999999996]
    ],
    [
        #gamma = 0.75
        [-18.048619027086353, -16.354914089347076, -20.098144329896904, -5.999999999999999, -13.893333333333327, -15.47199999999999, -5.999999999999998, -5.999999999999998, -5.999999999999998, -5.999999999999998],
        [-11.364085664816024, -6.829408299891047, -11.362958194659226, -5.999999999999998, -5.999999999999998, -5.999999999999998, -5.999999999999998, -5.999999999999998, -5.999999999999998, -5.999999999999998], 
        [-15.569230769230767, -11.999999999999996, -19.261538461538457, -11.999999999999996, -11.999999999999996, -11.999999999999996, -11.999999999999996, -11.999999999999996, -11.999999999999996, -11.999999999999996]
    ],
    [
        #gamma = 0.8
        [-22.14552188552189, -20.28181818181818, -24.856363636363632, -7.500000000000002, -16.749999999999993, -19.062499999999993, -7.5, -7.500000000000002, -7.500000000000002, -7.500000000000002], 
        [-13.227691215343736, -8.540503875101978, -13.175865235686418, -7.5, -7.500000000000001, -7.5, -7.499999999999998, -7.5, -7.5, -7.5], 
        [-18.625, -15.0, -22.375, -15.0, -15.0, -15.0, -15.0, -15.0, -15.0, -15.0]
    ],
    [
        #gamma = 0.85
        [-28.76278844268961, -26.61680527433105, -32.56131830251732, -9.999999999999993, -21.169811320754697, -24.752580989676016, -9.999999999999993, -9.999999999999993, -9.999999999999993, -9.999999999999993], 
        [-16.183356468130675, -11.33189687650437, -16.033301790463963, -9.999999999999993, -9.999999999999991, -9.999999999999993, -9.999999999999993, -9.999999999999993, -9.999999999999993, -9.999999999999993], 
        [-23.68253968253967, -19.999999999999986, -27.49206349206348, -19.999999999999986, -19.999999999999986, -19.999999999999986, -19.999999999999986, -19.999999999999986, -19.999999999999986, -19.999999999999986]
    ],
    [
        #gamma = 0.9
        [-41.27742867847752, -38.58932362753994, -47.172156505914224, -14.999999999999755, -29.095238095237843, -35.13605442176845, -14.999999999999753, -14.999999999999753, -14.999999999999753, -14.999999999999753], 
        [-21.789898957859354, -16.75709624029196, -21.448166972857727, -14.99999999999974, -14.99999999999974, -14.999999999999744, -14.999999999999735, -14.999999999999744, -14.999999999999744, -14.999999999999744], 
        [-33.74193548387047, -29.999999999999503, -37.61290322580595, -29.999999999999503, -29.999999999999503, -29.999999999999503, -29.999999999999503, -29.999999999999503, -29.999999999999503, -29.999999999999503]
    ],
    [
        #gamma = 0.95
        [-74.330382553884, -70.25959327963282, -85.68377649107512, -29.99999408538547, -49.09676827893381, -60.80124278465696, -29.999994085385474, -29.99999408538546, -29.999994085385453, -29.999994085385445], 
        [-37.67557701062915, -32.430971145564975, -36.94165998316571, -29.999994085385467, -29.999994085385474, -29.99999408538546, -29.999994085385474, -29.99999408538546, -29.999994085385453, -29.999994085385445], 
        [-63.80326685929545, -59.99998817077086, -67.73769308880364, -59.99998817077086, -59.99998817077086, -59.99998817077086, -59.99998817077086, -59.99998817077086, -59.99998817077086, -59.99998817077086]
    ]
])

strategies_new = np.array([
    [
        #gamma = 0.5
        [-9.971630613650078, -8.329547965975381, -10.400000000000002, -3.0, -7.145658263305322, -7.807570086858271, -3.0, -3.0, -3.0, -3.0],
        [-8.006725181677554, -3.239120519578889, -8.318825042314543, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0], 
        [-10.142857142857142, -6.0, -11.8, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0]
    ],
    [
        #gamma = 0.55
        [-11.021348325484793, -9.297315901165504, -11.555555555555557, -3.3333333333333344, -7.781785392245267, -8.62013208323949, -3.333333333333334, -3.333333333333334, -3.333333333333334, -3.333333333333334],
        [-8.587462664854904, -3.6318409401891993, -8.954863037552666, -3.3333333333333335, -3.333333333333333, -3.333333333333333, -3.333333333333334, -3.333333333333334, -3.333333333333334, -3.333333333333334], 
        [-10.995024875621892, -6.666666666666666, -13.11111111111111, -6.666666666666666, -6.666666666666666, -6.666666666666666, -6.666666666666666, -6.666666666666666, -6.666666666666666, -6.666666666666666]
    ],
    [
        #gamma = 0.6
        [-12.32898303784026, -10.502325751960964, -12.999999999999996, -3.75, -8.548962386511022, -9.61332369402514, -3.75, -3.75, -3.75, -3.75], 
        [-9.282266622638396, -4.123415277139301, -9.711631594171253, -3.75, -3.75, -3.75, -3.7499999999999982, -3.7499999999999982, -3.7499999999999982, -3.7499999999999982],
        [-12.03125, -7.499999999999999, -14.749999999999998, -7.499999999999999, -7.499999999999999, -7.499999999999999, -7.499999999999999, -7.499999999999999, -7.499999999999999, -7.499999999999999]
    ],
    [
        #gamma = 0.65
        [-14.003029931817217, -12.042950269851962, -14.857142857142856, -4.285714285714285, -9.495147583848745, -10.853891010078112, -4.285714285714285, -4.285714285714285, -4.285714285714285, -4.285714285714285],
        [-10.134043558251234, -4.755406686649815, -10.632410168304135, -4.285714285714285, -4.285714285714285, -4.285714285714285, -4.285714285714283, -4.285714285714285, -4.285714285714285, -4.285714285714285], 
        [-13.325526932084308, -8.57142857142857, -16.857142857142854, -8.57142857142857, -8.57142857142857, -8.57142857142857, -8.57142857142857, -8.57142857142857, -8.57142857142857, -8.57142857142857]
    ],
    [
        #gamma = 0.7
        [-16.222871953378966, -14.081267863466989, -17.33333333333332, -4.999999999999998, -10.696689761354886, -12.44648130468098, -4.999999999999998, -4.999999999999998, -4.999999999999998, -4.999999999999998], 
        [-11.202057922972564, -5.596353517389372, -11.786529452742403, -4.999999999999998, -4.999999999999998, -4.999999999999998, -4.999999999999998, -4.999999999999998, -4.999999999999998, -4.999999999999998], 
        [-14.999999999999998, -9.999999999999996, -19.666666666666657, -9.999999999999996, -9.999999999999996, -9.999999999999996, -9.999999999999996, -9.999999999999996, -9.999999999999996, -9.999999999999996]
    ],
    [
        #gamma = 0.75
        [-19.308479170159938, -16.90547338875094, -20.799999999999994, -5.999999999999998, -12.28450106157112, -14.566135204944077, -5.999999999999998, -5.999999999999998, -5.999999999999998, -5.999999999999998],
        [-12.544796172251527, -6.768001890012002, -13.294099856705948, -5.999999999999998, -5.999999999999998, -5.999999999999998, -5.999999999999998, -5.999999999999998, -5.999999999999998, -5.999999999999998],
        [-17.272727272727266, -11.999999999999996, -22.727272727272727, -11.999999999999996, -11.999999999999996, -11.999999999999996, -11.999999999999996, -11.999999999999996, -11.999999999999996, -11.999999999999996]
    ],
    [
        #gamma = 0.8
        [-23.893457253646822, -21.08473359184259, -25.999999999999993, -7.5, -14.507575757575752, -17.533574380165284, -7.5, -7.5, -7.5, -7.5], 
        [-14.490509238825954, -8.509860614636768, -15.386958983336473, -7.5, -7.5, -7.5, -7.499999999999998, -7.499999999999998, -7.499999999999998, -7.499999999999998],
        [-20.576923076923077, -15.0, -26.346153846153843, -15.0, -15.0, -15.0, -15.0, -15.0, -15.0, -15.0]
    ],
    [
        #gamma = 0.85
        [-31.441011928070928, -27.93074059489324, -34.66666666666662, -9.999999999999995, -17.918673087212397, -22.024180963889137, -9.999999999999993, -9.999999999999993, -9.999999999999993, -9.999999999999993],
        [-17.639003916040554, -11.393751484909842, -18.76767423832613, -9.999999999999993, -9.999999999999993, -9.999999999999993, -9.999999999999993, -9.999999999999993, -9.999999999999993, -9.999999999999993],
        [-25.918367346938766, -19.999999999999986, -32.0408163265306, -19.999999999999986, -19.999999999999986, -19.999999999999986, -19.999999999999986, -19.999999999999986, -19.999999999999986, -19.999999999999986]
    ],
    [
        #gamma = 0.9
        [-46.29771063517065, -41.3346069287458, -51.99999999999911, -14.999999999999758, -24.102091020909963, -29.845476868052558, -14.999999999999753, -14.999999999999755, -14.999999999999758, -14.999999999999758],
        [-23.67768368500686, -17.045644330889637, -24.990414538712052, -14.999999999999744, -14.999999999999744, -14.999999999999744, -14.999999999999735, -14.999999999999737, -14.999999999999744, -14.999999999999744],
        [-36.304347826086456, -29.999999999999503, -42.82608695652125, -29.999999999999503, -29.999999999999503, -29.999999999999503, -29.999999999999503, -29.999999999999503, -29.999999999999503, -29.999999999999503]
    ],
    [
        #gamma = 0.95
        [-90.02353744521069, -80.58330841514423, -103.99997949600291, -29.999994085385467, -40.701367910403526, -49.08140251090573, -29.999994085385474, -29.999994085385474, -29.99999408538546, -29.99999408538546],
        [-40.98021846017648, -33.30390039789375, -41.67151145465002, -29.99999408538546, -29.99999408538546, -29.99999408538546, -29.999994085385474, -29.999994085385467, -29.99999408538546, -29.99999408538546],
        [-66.74417421728249, -59.99998817077086, -73.720918403329, -59.99998817077086, -59.99998817077086, -59.99998817077086, -59.99998817077086, -59.99998817077086, -59.99998817077086, -59.99998817077086]
    ]
])

strategies_new_hp = np.array([
    [
        #gamma = 0.5
        [-9.719173540976024, -7.609338781384113, -9.400000000000002, -3.0, -6.5854341736694675, -7.157898453499047, -3.0, -3.0, -3.0, -3.0],
        [-7.268823723245133, -2.1730810644505407, -7.549757772166615, -3.0, -2.045087267133338, -2.002079646956825, -3.0, -3.0, -3.0, -3.0],
        [-9.571428571428573, -4.0, -11.8, -6.0, -4.0, -4.0, -6.0, -6.0, -6.0, -6.0]
    ],
    [
        #gamma = 0.55
        [-10.699273258735658, -8.491372310917912, -10.444444444444446, -3.3333333333333344, -7.180643222122033, -7.905699819738657, -3.333333333333334, -3.333333333333334, -3.333333333333334, -3.333333333333334],
        [-7.746606127473077, -2.43992565998048, -8.081471902098858, -3.3333333333333335, -2.2825737939584805, -2.225591530098113, -3.333333333333334, -3.333333333333334, -3.333333333333334, -3.333333333333334], 
        [-10.265339966832505, -4.444444444444445, -13.11111111111111, -6.666666666666666, -4.444444444444445, -4.444444444444445, -6.666666666666666, -6.666666666666666, -6.666666666666666, -6.666666666666666]
    ],
    [
        #gamma = 0.6
        [-11.915929662416103, -9.589849298993267, -11.75, -3.75, -7.900453955901424, -8.820982654292013, -3.75, -3.75, -3.75, -3.75], 
        [-8.31035735439421, -2.7749916598389284, -8.70768658523047, -3.75, -2.5818683147693022, -2.5055426385408794, -3.7499999999999982, -3.7499999999999982, -3.7499999999999982, -3.7499999999999982],
        [-11.093749999999998, -4.999999999999998, -14.749999999999998, -7.499999999999999, -4.999999999999998, -4.999999999999998, -7.499999999999999, -7.499999999999999, -7.499999999999999, -7.499999999999999]
    ],
    [
        #gamma = 0.65
        [-13.46808928556185, -10.994675136860383, -13.428571428571425, -4.285714285714285, -8.791170111127872, -9.966299560839758, -4.285714285714285, -4.285714285714285, -4.285714285714285, -4.285714285714285],
        [-8.991119039272863, -3.207519570407403, -9.46118048970386, -4.285714285714285, -2.9704546395110487, -2.866499182573401, -4.285714285714283, -4.285714285714285, -4.285714285714285, -4.285714285714285], 
        [-12.107728337236535, -5.7142857142857135, -16.857142857142854, -8.57142857142857, -5.7142857142857135, -5.7142857142857135, -8.57142857142857, -8.57142857142857, -8.57142857142857, -8.57142857142857]
    ],
    [
        #gamma = 0.7
        [-15.519275765436088, -12.85406950353902, -15.66666666666666, -4.999999999999998, -9.926866820631252, -11.44020004729166, -4.999999999999998, -4.999999999999998, -4.999999999999998, -4.999999999999998], 
        [-9.839250275972276, -3.7861682757442274, -10.394471015919759, -4.999999999999998, -3.494773403084623, -3.349758746975971, -4.999999999999998, -4.999999999999998, -4.999999999999998, -4.999999999999998], 
        [-13.390804597701147, -6.666666666666664, -18.563218390804597, -9.999999999999996, -6.666666666666664, -6.666666666666664, -9.999999999999996, -9.999999999999996, -9.999999999999996, -9.999999999999996]
    ],
    [
        #gamma = 0.75
        [-18.36109986884447, -15.43176076864946, -18.79999999999999, -5.999999999999998, -11.435244161358806, -13.40854936643812, -5.999999999999998, -5.999999999999998, -5.999999999999998, -5.999999999999998],
        [-10.943475289847093, -4.598413846031705, -11.598975150839017, -5.999999999999998, -4.239789828873535, -4.030553186462099, -5.999999999999998, -5.999999999999998, -5.999999999999998, -5.999999999999998],
        [-15.090909090909092, -7.999999999999998, -20.54545454545454, -11.999999999999996, -7.999999999999998, -7.999999999999998, -11.999999999999996, -11.999999999999996, -11.999999999999996, -11.999999999999996]
    ],
    [
        #gamma = 0.8
        [-22.570982335200334, -19.24895878213413, -23.499999999999993, -7.5, -13.560606060606055, -16.177685950413217, -7.5, -7.5, -7.5, -7.5], 
        [-12.478812176990406, -5.819192227877442, -13.253179165148516, -7.5, -5.3788063991689485, -5.061976749922556, -7.499999999999998, -7.499999999999998, -7.499999999999998, -7.499999999999998],
        [-17.5, -9.999999999999996, -23.269230769230763, -15.0, -9.999999999999996, -9.999999999999996, -15.0, -15.0, -15.0, -15.0]
    ],
    [
        #gamma = 0.85
        [-29.482545019887265, -25.5076675415293, -31.333333333333307, -9.999999999999995, -16.848582129480995, -20.39929164444466, -9.999999999999993, -9.999999999999993, -9.999999999999993, -9.999999999999993],
        [-14.787244539492471, -7.857939347618573, -15.769436374940053, -9.999999999999993, -7.32616777361741, -6.810790171605226, -9.999999999999993, -9.999999999999993, -9.999999999999993, -9.999999999999993],
        [-21.29251700680272, -13.333333333333327, -27.414965986394556, -19.999999999999986, -13.333333333333327, -13.333333333333327, -19.999999999999986, -19.999999999999986, -19.999999999999986, -19.999999999999986]
    ],
    [
        #gamma = 0.9
        [-43.05838928896747, -37.775876262699036, -46.9999999999992, -14.999999999999758, -22.87207872078696, -27.839331345342718, -14.999999999999753, -14.999999999999755, -14.999999999999758, -14.999999999999758],
        [-19.46413862547111, -11.956802350913414, -20.39828021818137, -14.999999999999744, -11.364732485804135, -10.425642784865175, -14.999999999999735, -14.999999999999737, -14.999999999999744, -14.999999999999744],
        [-28.478260869564892, -19.99999999999967, -34.99999999999967, -29.999999999999503, -19.99999999999967, -19.99999999999967, -29.999999999999503, -29.999999999999503, -29.999999999999503, -29.999999999999503]
    ],
    [
        #gamma = 0.95
        [-82.95936829917181, -73.7477254632255, -93.9999815713063, -29.999994085385467, -39.25523631242812, -46.50283380475434, -29.999994085385474, -29.999994085385474, -29.99999408538546, -29.99999408538546],
        [-33.19364521856391, -24.56053834756213, -33.7549493800915, -29.99999408538546, -24.202678066901264, -22.116918703241936, -29.999994085385474, -29.999994085385467, -29.99999408538546, -29.99999408538546],
        [-49.06975955570771, -39.99999211384725, -56.046503741754215, -59.99998817077086, -39.99999211384725, -39.99999211384725, -59.99998817077086, -59.99998817077086, -59.99998817077086, -59.99998817077086]
    ]
])

optimal_mixed_gamma = np.array([
    [
        #gamma = 0.5
        [-7.378487640724603, -3.3197512739857147, -7.496142688168831, -3.0000000000000004, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0], 
        [-8.006725181677554, -3.239120519578889, -8.318825042314543, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0],
        [-7.268823723245133, -2.1730810644505407, -7.549757772166615, -3.0, -2.045087267133338, -2.002079646956825, -3.0, -3.0, -3.0, -3.0]
    ],
    [
        #gamma = 0.55
        [-7.853847828218031, -3.71793284163171, -7.966237505840968, -3.333333333333334, -3.333333333333333, -3.333333333333333, -3.333333333333334, -3.333333333333334, -3.333333333333334, -3.333333333333334],
        [-8.587462664854904, -3.6318409401891993, -8.954863037552666, -3.3333333333333335, -3.333333333333333, -3.333333333333333, -3.333333333333334, -3.333333333333334, -3.333333333333334, -3.333333333333334],
        [-7.746606127473077, -2.43992565998048, -8.081471902098858, -3.3333333333333335, -2.2825737939584805, -2.225591530098113, -3.333333333333334, -3.333333333333334, -3.333333333333334, -3.333333333333334]
    ],
    [
        #gamma = 0.6
        [-8.42965572232598, -4.21204039850597, -8.528273788767603, -3.7499999999999982, -3.7499999999999982, -3.7499999999999982, -3.7499999999999982, -3.7499999999999982, -3.7499999999999982, -3.7499999999999982],
        [-9.282266622638396, -4.123415277139301, -9.711631594171253, -3.75, -3.75, -3.75, -3.7499999999999982, -3.7499999999999982, -3.7499999999999982, -3.7499999999999982],
        [-8.31035735439421, -2.7749916598389284, -8.70768658523047, -3.75, -2.5818683147693022, -2.5055426385408794, -3.7499999999999982, -3.7499999999999982, -3.7499999999999982, -3.7499999999999982]
    ],
    [
        #gamma = 0.65
        [-9.145905020699661, -4.841618667247477, -9.218958552423087, -4.285714285714285, -4.285714285714285, -4.285714285714285, -4.285714285714283, -4.285714285714283, -4.285714285714283, -4.285714285714283],
        [-10.134043558251234, -4.755406686649815, -10.632410168304135, -4.285714285714285, -4.285714285714285, -4.285714285714285, -4.285714285714283, -4.285714285714285, -4.285714285714285, -4.285714285714285],
        [-8.991119039272863, -3.207519570407403, -9.46118048970386, -4.285714285714285, -2.9704546395110487, -2.866499182573401, -4.285714285714283, -4.285714285714285, -4.285714285714285, -4.285714285714285]
    ],
    [
        #gamma = 0.7
        [-10.068193838520614, -5.671752735193775, -10.099054988853647, -4.999999999999998, -4.999999999999998, -4.999999999999998, -4.999999999999998, -4.999999999999998, -4.999999999999998, -4.999999999999998],
        [-11.202057922972564, -5.596353517389372, -11.786529452742403, -4.999999999999998, -4.999999999999998, -4.999999999999998, -4.999999999999998, -4.999999999999998, -4.999999999999998, -4.999999999999998], 
        [-9.839250275972276, -3.7861682757442274, -10.394471015919759, -4.999999999999998, -3.494773403084623, -3.349758746975971, -4.999999999999998, -4.999999999999998, -4.999999999999998, -4.999999999999998],
    ],
    [
        #gamma = 0.75
        [-11.364085664816024, -6.829408299891047, -11.362958194659226, -5.999999999999998, -5.999999999999998, -5.999999999999998, -5.999999999999998, -5.999999999999998, -5.999999999999998, -5.999999999999998],
        [-12.544796172251527, -6.768001890012002, -13.294099856705948, -5.999999999999998, -5.999999999999998, -5.999999999999998, -5.999999999999998, -5.999999999999998, -5.999999999999998, -5.999999999999998],
        [-10.943475289847093, -4.598413846031705, -11.598975150839017, -5.999999999999998, -4.239789828873535, -4.030553186462099, -5.999999999999998, -5.999999999999998, -5.999999999999998, -5.999999999999998]
    ],
    [
        #gamma = 0.8
        [-13.227691215343736, -8.540503875101978, -13.175865235686418, -7.5, -7.500000000000001, -7.5, -7.499999999999998, -7.5, -7.5, -7.5], 
        [-14.490509238825954, -8.509860614636768, -15.386958983336473, -7.5, -7.5, -7.5, -7.499999999999998, -7.499999999999998, -7.499999999999998, -7.499999999999998],
        [-12.478812176990406, -5.819192227877442, -13.253179165148516, -7.5, -5.3788063991689485, -5.061976749922556, -7.499999999999998, -7.499999999999998, -7.499999999999998, -7.499999999999998]
    ],
    [
        #gamma = 0.85
        [-16.183356468130675, -11.33189687650437, -16.033301790463963, -9.999999999999993, -9.999999999999991, -9.999999999999993, -9.999999999999993, -9.999999999999993, -9.999999999999993, -9.999999999999993],
        [-17.639003916040554, -11.393751484909842, -18.76767423832613, -9.999999999999993, -9.999999999999993, -9.999999999999993, -9.999999999999993, -9.999999999999993, -9.999999999999993, -9.999999999999993],
        [-14.787244539492471, -7.857939347618573, -15.769436374940053, -9.999999999999993, -7.32616777361741, -6.810790171605226, -9.999999999999993, -9.999999999999993, -9.999999999999993, -9.999999999999993]
    ],
    [
        #gamma = 0.9
        [-21.789898957859354, -16.75709624029196, -21.448166972857727, -14.99999999999974, -14.99999999999974, -14.999999999999744, -14.999999999999735, -14.999999999999744, -14.999999999999744, -14.999999999999744],
        [-23.67768368500686, -17.045644330889637, -24.990414538712052, -14.999999999999744, -14.999999999999744, -14.999999999999744, -14.999999999999735, -14.999999999999737, -14.999999999999744, -14.999999999999744],
        [-19.46413862547111, -11.956802350913414, -20.39828021818137, -14.999999999999744, -11.364732485804135, -10.425642784865175, -14.999999999999735, -14.999999999999737, -14.999999999999744, -14.999999999999744]
    ],
    [
        #gamma = 0.95
        [-37.67557701062915, -32.430971145564975, -36.94165998316571, -29.999994085385467, -29.999994085385474, -29.99999408538546, -29.999994085385474, -29.99999408538546, -29.999994085385453, -29.999994085385445],
        [-40.98021846017648, -33.30390039789375, -41.67151145465002, -29.99999408538546, -29.99999408538546, -29.99999408538546, -29.999994085385474, -29.999994085385467, -29.99999408538546, -29.99999408538546],
        [-33.19364521856391, -24.56053834756213, -33.7549493800915, -29.99999408538546, -24.202678066901264, -22.116918703241936, -29.999994085385474, -29.999994085385467, -29.99999408538546, -29.99999408538546]
    ]
])

gamma = np.array([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95])

#Plot 1

V0_INIT = []
V0_UPD1 = []
V0_UPD2 = []

strategies = optimal_mixed_gamma

for i in range(len(strategies)):
    V0_INIT.append(strategies[i][0][0])
    V0_UPD1.append(strategies[i][1][0])
    V0_UPD2.append(strategies[i][2][0])


plt.plot(gamma, np.asarray(V0_INIT), marker='o')
plt.plot(gamma, np.asarray(V0_UPD1), marker='x')
plt.plot(gamma, np.asarray(V0_UPD2), marker='+')
plt.ylabel("Defender's Utility $\longrightarrow$")
plt.xlabel("$\gamma \longrightarrow$")
plt.title("Defender's value in state S0.")

plt.legend(['Initial Model', 'Updated model w/ tuned params', 'Updated model w/ non-uniform costs'], loc='lower left')

plt.show()

#Plot 2

V1_INIT = []
V1_UPD1 = []
V1_UPD2 = []

for i in range(len(strategies)):
    V1_INIT.append(strategies[i][0][1])
    V1_UPD1.append(strategies[i][1][1])
    V1_UPD2.append(strategies[i][2][1])


plt.plot(gamma, np.asarray(V1_INIT), marker='o')
plt.plot(gamma, np.asarray(V1_UPD1), marker='x')
plt.plot(gamma, np.asarray(V1_UPD2), marker='+')
plt.ylabel("Defender's Utility $\longrightarrow$")
plt.xlabel("$\gamma \longrightarrow$")
plt.title("Defender's value in state S1.")

plt.legend(['Initial Model', 'Updated model w/ tuned params', 'Updated model w/ non-uniform costs'], loc='lower left')

plt.show()

#Plot 3

V2_INIT = []
V2_UPD1 = []
V2_UPD2 = []

for i in range(len(strategies)):
    V2_INIT.append(strategies[i][0][2])
    V2_UPD1.append(strategies[i][1][2])
    V2_UPD2.append(strategies[i][2][2])


plt.plot(gamma, np.asarray(V2_INIT), marker='o')
plt.plot(gamma, np.asarray(V2_UPD1), marker='x')
plt.plot(gamma, np.asarray(V2_UPD2), marker='+')
plt.ylabel("Defender's Utility $\longrightarrow$")
plt.xlabel("$\gamma \longrightarrow$")
plt.title("Defender's value in state S2.")

plt.legend(['Initial Model', 'Updated model w/ tuned params', 'Updated model w/ non-uniform costs'], loc='lower left')

plt.show()

#Plot 4

V3_INIT = []
V3_UPD1 = []
V3_UPD2 = []

for i in range(len(strategies)):
    V3_INIT.append(strategies[i][0][3])
    V3_UPD1.append(strategies[i][1][3])
    V3_UPD2.append(strategies[i][2][3])


plt.plot(gamma, np.asarray(V3_INIT), marker='o')
plt.plot(gamma, np.asarray(V3_UPD1), marker='x')
plt.plot(gamma, np.asarray(V3_UPD2), marker='+')
plt.ylabel("Defender's Utility $\longrightarrow$")
plt.xlabel("$\gamma \longrightarrow$")
plt.title("Defender's value in state S3.")

plt.legend(['Initial Model', 'Updated model w/ tuned params', 'Updated model w/ non-uniform costs'], loc='lower left')

plt.show()

#Plot 5

V4_INIT = []
V4_UPD1 = []
V4_UPD2 = []

for i in range(len(strategies)):
    V4_INIT.append(strategies[i][0][4])
    V4_UPD1.append(strategies[i][1][4])
    V4_UPD2.append(strategies[i][2][4])


plt.plot(gamma, np.asarray(V4_INIT), marker='o')
plt.plot(gamma, np.asarray(V4_UPD1), marker='x')
plt.plot(gamma, np.asarray(V4_UPD2), marker='+')
plt.ylabel("Defender's Utility $\longrightarrow$")
plt.xlabel("$\gamma \longrightarrow$")
plt.title("Defender's value in state S4.")

plt.legend(['Initial Model', 'Updated model w/ tuned params', 'Updated model w/ non-uniform costs'], loc='lower left')

plt.show()

#Plot 6

V5_INIT = []
V5_UPD1 = []
V5_UPD2 = []

for i in range(len(strategies)):
    V5_INIT.append(strategies[i][0][5])
    V5_UPD1.append(strategies[i][1][5])
    V5_UPD2.append(strategies[i][2][5])


plt.plot(gamma, np.asarray(V5_INIT), marker='o')
plt.plot(gamma, np.asarray(V5_UPD1), marker='x')
plt.plot(gamma, np.asarray(V5_UPD2), marker='+')
plt.ylabel("Defender's Utility $\longrightarrow$")
plt.xlabel("$\gamma \longrightarrow$")
plt.title("Defender's value in state S5.")

plt.legend(['Initial Model', 'Updated model w/ tuned params', 'Updated model w/ non-uniform costs'], loc='lower left')

plt.show()

#Plot 7

V6_INIT = []
V6_UPD1 = []
V6_UPD2 = []

for i in range(len(strategies)):
    V6_INIT.append(strategies[i][0][6])
    V6_UPD1.append(strategies[i][1][6])
    V6_UPD2.append(strategies[i][2][6])


plt.plot(gamma, np.asarray(V6_INIT), marker='o')
plt.plot(gamma, np.asarray(V6_UPD1), marker='x')
plt.plot(gamma, np.asarray(V6_UPD2), marker='+')
plt.ylabel("Defender's Utility $\longrightarrow$")
plt.xlabel("$\gamma \longrightarrow$")
plt.title("Defender's value in state S6.")

plt.legend(['Initial Model', 'Updated model w/ tuned params', 'Updated model w/ non-uniform costs'], loc='lower left')

plt.show()

#Plot 8

V7_INIT = []
V7_UPD1 = []
V7_UPD2 = []

for i in range(len(strategies)):
    V7_INIT.append(strategies[i][0][7])
    V7_UPD1.append(strategies[i][1][7])
    V7_UPD2.append(strategies[i][2][7])


plt.plot(gamma, np.asarray(V7_INIT), marker='o')
plt.plot(gamma, np.asarray(V7_UPD1), marker='x')
plt.plot(gamma, np.asarray(V7_UPD2), marker='+')
plt.ylabel("Defender's Utility $\longrightarrow$")
plt.xlabel("$\gamma \longrightarrow$")
plt.title("Defender's value in state S7.")

plt.legend(['Initial Model', 'Updated model w/ tuned params', 'Updated model w/ non-uniform costs'], loc='lower left')

plt.show()

#Plot 9

V8_INIT = []
V8_UPD1 = []
V8_UPD2 = []

for i in range(len(strategies)):
    V8_INIT.append(strategies[i][0][8])
    V8_UPD1.append(strategies[i][1][8])
    V8_UPD2.append(strategies[i][2][8])


plt.plot(gamma, np.asarray(V8_INIT), marker='o')
plt.plot(gamma, np.asarray(V8_UPD1), marker='x')
plt.plot(gamma, np.asarray(V8_UPD2), marker='+')
plt.ylabel("Defender's Utility $\longrightarrow$")
plt.xlabel("$\gamma \longrightarrow$")
plt.title("Defender's value in state S8.")

plt.legend(['Initial Model', 'Updated model w/ tuned params', 'Updated model w/ non-uniform costs'], loc='lower left')

plt.show()

#Plot 10

V9_INIT = []
V9_UPD1 = []
V9_UPD2 = []

for i in range(len(strategies)):
    V9_INIT.append(strategies[i][0][9])
    V9_UPD1.append(strategies[i][1][9])
    V9_UPD2.append(strategies[i][2][9])


plt.plot(gamma, np.asarray(V9_INIT), marker='o')
plt.plot(gamma, np.asarray(V9_UPD1), marker='x')
plt.plot(gamma, np.asarray(V9_UPD2), marker='+')
plt.ylabel("Defender's Utility $\longrightarrow$")
plt.xlabel("$\gamma \longrightarrow$")
plt.title("Defender's value in state S9.")

plt.legend(['Initial Model', 'Updated model w/ tuned params', 'Updated model w/ non-uniform costs'], loc='lower left')

plt.show()