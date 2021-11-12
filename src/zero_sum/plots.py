import matplotlib.pyplot as plt
import numpy as np

strategies = np.array([
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

gamma = np.array([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95])

#Plot 1

V0_URS = []
V0_OPT = []
V0_MMP = []

for i in range(len(strategies)):
    V0_URS.append(strategies[i][0][0])
    V0_OPT.append(strategies[i][1][0])
    V0_MMP.append(strategies[i][2][0])


plt.plot(gamma, np.asarray(V0_URS), marker='o')
plt.plot(gamma, np.asarray(V0_OPT), marker='x')
plt.plot(gamma, np.asarray(V0_MMP), marker='+')
plt.ylabel("Defender's Utility $\longrightarrow$")
plt.xlabel("$\gamma \longrightarrow$")
plt.title("Defender's value in state S0.")

plt.legend(['Uniform Random Strategy', 'Optimal Mixed Strategy', 'Min Max Pure Strategy'], loc='lower left')

plt.show()

#Plot 2

V1_URS = []
V1_OPT = []
V1_MMP = []

for i in range(len(strategies)):
    V1_URS.append(strategies[i][0][1])
    V1_OPT.append(strategies[i][1][1])
    V1_MMP.append(strategies[i][2][1])


plt.plot(gamma, np.asarray(V1_URS), marker='o')
plt.plot(gamma, np.asarray(V1_OPT), marker='x')
plt.plot(gamma, np.asarray(V1_MMP), marker='+')
plt.ylabel("Defender's Utility $\longrightarrow$")
plt.xlabel("$\gamma \longrightarrow$")
plt.title("Defender's value in state S1.")

plt.legend(['Uniform Random Strategy', 'Optimal Mixed Strategy', 'Min Max Pure Strategy'], loc='lower left')

plt.show()

#Plot 3

V2_URS = []
V2_OPT = []
V2_MMP = []

for i in range(len(strategies)):
    V2_URS.append(strategies[i][0][2])
    V2_OPT.append(strategies[i][1][2])
    V2_MMP.append(strategies[i][2][2])


plt.plot(gamma, np.asarray(V2_URS), marker='o')
plt.plot(gamma, np.asarray(V2_OPT), marker='x')
plt.plot(gamma, np.asarray(V2_MMP), marker='+')
plt.ylabel("Defender's Utility $\longrightarrow$")
plt.xlabel("$\gamma \longrightarrow$")
plt.title("Defender's value in state S2.")

plt.legend(['Uniform Random Strategy', 'Optimal Mixed Strategy', 'Min Max Pure Strategy'], loc='lower left')

plt.show()

#Plot 4

V3_URS = []
V3_OPT = []
V3_MMP = []

for i in range(len(strategies)):
    V3_URS.append(strategies[i][0][3])
    V3_OPT.append(strategies[i][1][3])
    V3_MMP.append(strategies[i][2][3])


plt.plot(gamma, np.asarray(V3_URS), marker='o')
plt.plot(gamma, np.asarray(V3_OPT), marker='x')
plt.plot(gamma, np.asarray(V3_MMP), marker='+')
plt.ylabel("Defender's Utility $\longrightarrow$")
plt.xlabel("$\gamma \longrightarrow$")
plt.title("Defender's value in state S3.")

plt.legend(['Uniform Random Strategy', 'Optimal Mixed Strategy', 'Min Max Pure Strategy'], loc='lower left')

plt.show()

#Plot 5

V4_URS = []
V4_OPT = []
V4_MMP = []

for i in range(len(strategies)):
    V4_URS.append(strategies[i][0][4])
    V4_OPT.append(strategies[i][1][4])
    V4_MMP.append(strategies[i][2][4])


plt.plot(gamma, np.asarray(V4_URS), marker='o')
plt.plot(gamma, np.asarray(V4_OPT), marker='x')
plt.plot(gamma, np.asarray(V4_MMP), marker='+')
plt.ylabel("Defender's Utility $\longrightarrow$")
plt.xlabel("$\gamma \longrightarrow$")
plt.title("Defender's value in state S4.")

plt.legend(['Uniform Random Strategy', 'Optimal Mixed Strategy', 'Min Max Pure Strategy'], loc='lower left')

plt.show()

#Plot 6

V5_URS = []
V5_OPT = []
V5_MMP = []

for i in range(len(strategies)):
    V5_URS.append(strategies[i][0][5])
    V5_OPT.append(strategies[i][1][5])
    V5_MMP.append(strategies[i][2][5])


plt.plot(gamma, np.asarray(V5_URS), marker='o')
plt.plot(gamma, np.asarray(V5_OPT), marker='x')
plt.plot(gamma, np.asarray(V5_MMP), marker='+')
plt.ylabel("Defender's Utility $\longrightarrow$")
plt.xlabel("$\gamma \longrightarrow$")
plt.title("Defender's value in state S5.")

plt.legend(['Uniform Random Strategy', 'Optimal Mixed Strategy', 'Min Max Pure Strategy'], loc='lower left')

plt.show()

#Plot 7

V6_URS = []
V6_OPT = []
V6_MMP = []

for i in range(len(strategies)):
    V6_URS.append(strategies[i][0][6])
    V6_OPT.append(strategies[i][1][6])
    V6_MMP.append(strategies[i][2][6])


plt.plot(gamma, np.asarray(V6_URS), marker='o')
plt.plot(gamma, np.asarray(V6_OPT), marker='x')
plt.plot(gamma, np.asarray(V6_MMP), marker='+')
plt.ylabel("Defender's Utility $\longrightarrow$")
plt.xlabel("$\gamma \longrightarrow$")
plt.title("Defender's value in state S6.")

plt.legend(['Uniform Random Strategy', 'Optimal Mixed Strategy', 'Min Max Pure Strategy'], loc='lower left')

plt.show()

#Plot 8

V7_URS = []
V7_OPT = []
V7_MMP = []

for i in range(len(strategies)):
    V7_URS.append(strategies[i][0][7])
    V7_OPT.append(strategies[i][1][7])
    V7_MMP.append(strategies[i][2][7])


plt.plot(gamma, np.asarray(V7_URS), marker='o')
plt.plot(gamma, np.asarray(V7_OPT), marker='x')
plt.plot(gamma, np.asarray(V7_MMP), marker='+')
plt.ylabel("Defender's Utility $\longrightarrow$")
plt.xlabel("$\gamma \longrightarrow$")
plt.title("Defender's value in state S7.")

plt.legend(['Uniform Random Strategy', 'Optimal Mixed Strategy', 'Min Max Pure Strategy'], loc='lower left')

plt.show()

#Plot 9

V8_URS = []
V8_OPT = []
V8_MMP = []

for i in range(len(strategies)):
    V8_URS.append(strategies[i][0][8])
    V8_OPT.append(strategies[i][1][8])
    V8_MMP.append(strategies[i][2][8])


plt.plot(gamma, np.asarray(V8_URS), marker='o')
plt.plot(gamma, np.asarray(V8_OPT), marker='x')
plt.plot(gamma, np.asarray(V8_MMP), marker='+')
plt.ylabel("Defender's Utility $\longrightarrow$")
plt.xlabel("$\gamma \longrightarrow$")
plt.title("Defender's value in state S8.")

plt.legend(['Uniform Random Strategy', 'Optimal Mixed Strategy', 'Min Max Pure Strategy'], loc='lower left')

plt.show()

#Plot 10

V9_URS = []
V9_OPT = []
V9_MMP = []

for i in range(len(strategies)):
    V9_URS.append(strategies[i][0][9])
    V9_OPT.append(strategies[i][1][9])
    V9_MMP.append(strategies[i][2][9])


plt.plot(gamma, np.asarray(V9_URS), marker='o')
plt.plot(gamma, np.asarray(V9_OPT), marker='x')
plt.plot(gamma, np.asarray(V9_MMP), marker='+')
plt.ylabel("Defender's Utility $\longrightarrow$")
plt.xlabel("$\gamma \longrightarrow$")
plt.title("Defender's value in state S9.")

plt.legend(['Uniform Random Strategy', 'Optimal Mixed Strategy', 'Min Max Pure Strategy'], loc='lower left')

plt.show()