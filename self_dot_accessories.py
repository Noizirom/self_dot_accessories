import bpy
import numpy as np
import os
import sys
from addon_utils import check, paths, enable
from mathutils import Matrix, Vector, kdtree



############################################################################################################

MR_list = [
        "spine.006",
        "spine.005",
        "shoulder.L",
        "shoulder.R",
        "breast.L",
        "breast.R",
        "spine.003",
        "spine.002",
        "spine.001",
        "spine",
        "thigh.L",
        "shin.L",
        "foot.L",
        "toe.L",
        "thigh.R",
        "shin.R",
        "foot.R",
        "toe.R",
        "upper_arm.L",
        "forearm.L",
        "hand.L",
        "upper_arm.R",
        "forearm.R",
        "hand.R"
        ]

MR_face = ['face', 'nose', 'nose.001', 'nose.002',
       'nose.003', 'nose.004', 'lip.T.L', 'lip.T.L.001', 'lip.B.L',
       'lip.B.L.001', 'jaw', 'chin', 'chin.001', 'ear.L', 'ear.L.001',
       'ear.L.002', 'ear.L.003', 'ear.L.004', 'ear.R', 'ear.R.001',
       'ear.R.002', 'ear.R.003', 'ear.R.004', 'lip.T.R', 'lip.T.R.001',
       'lip.B.R', 'lip.B.R.001', 'brow.B.L', 'brow.B.L.001',
       'brow.B.L.002', 'brow.B.L.003', 'lid.T.L', 'lid.T.L.001',
       'lid.T.L.002', 'lid.T.L.003', 'lid.B.L', 'lid.B.L.001',
       'lid.B.L.002', 'lid.B.L.003', 'brow.B.R', 'brow.B.R.001',
       'brow.B.R.002', 'brow.B.R.003', 'lid.T.R', 'lid.T.R.001',
       'lid.T.R.002', 'lid.T.R.003', 'lid.B.R', 'lid.B.R.001',
       'lid.B.R.002', 'lid.B.R.003', 'forehead.L', 'forehead.L.001',
       'forehead.L.002', 'temple.L', 'jaw.L', 'jaw.L.001', 'chin.L',
       'cheek.B.L', 'cheek.B.L.001', 'brow.T.L', 'brow.T.L.001',
       'brow.T.L.002', 'brow.T.L.003', 'forehead.R', 'forehead.R.001',
       'forehead.R.002', 'temple.R', 'jaw.R', 'jaw.R.001', 'chin.R',
       'cheek.B.R', 'cheek.B.R.001', 'brow.T.R', 'brow.T.R.001',
       'brow.T.R.002', 'brow.T.R.003', 'eye.L', 'eye.R', 'cheek.T.L',
       'cheek.T.L.001', 'nose.L', 'nose.L.001', 'cheek.T.R',
       'cheek.T.R.001', 'nose.R', 'nose.R.001', 'teeth.T', 'teeth.B',
       'tongue', 'tongue.001', 'tongue.002']

#left hand
thumb_L = ["thumb.01.L", "thumb.02.L", "thumb.03.L"]
index_L = ["palm.01.L", "f_index.01.L", "f_index.02.L", "f_index.03.L"]
middle_L = ["palm.02.L", "f_middle.01.L", "f_middle.02.L", "f_middle.03.L"]
ring_L = ["palm.03.L", "f_ring.01.L", "f_ring.02.L", "f_ring.03.L"]
pinky_L = ["palm.04.L", "f_pinky.01.L", "f_pinky.02.L", "f_pinky.03.L"]
#right hand
thumb_R = ["thumb.01.R", "thumb.02.R", "thumb.03.R"]
index_R = ["palm.01.R", "f_index.01.R", "f_index.02.R", "f_index.03.R"]
middle_R = ["palm.02.R", "f_middle.01.R", "f_middle.02.R", "f_middle.03.R"]
ring_R = ["palm.03.R", "f_ring.01.R", "f_ring.02.R", "f_ring.03.R"]
pinky_R = ["palm.04.R", "f_pinky.01.R", "f_pinky.02.R", "f_pinky.03.R"]

MR_fingers = thumb_L + index_L + middle_L + ring_L + pinky_L + thumb_R + index_R + middle_R + ring_R + pinky_R

############################################################################################################

MB_list = [
        "head",
        "neck",
        "clavicle_L",
        "clavicle_R",
        "breast_L",
        "breast_R",
        "spine03",
        "spine02",
        "spine01",
        "pelvis",
        "thigh_L",
        "calf_L",
        "foot_L",
        "toes_L",
        "thigh_R",
        "calf_R",
        "foot_R",
        "toes_R",
        "upperarm_L",
        "lowerarm_L",
        "hand_L",
        "upperarm_R",
        "lowerarm_R",
        "hand_R"
        ]

#left hand
thumb_l = ["thumb01_L", "thumb02_L", "thumb03_L"]
index_l = ["index00_L", "index01_L", "index02_L", "index03_L"]
middle_l = ["middle00_L", "middle01_L", "middle02_L", "middle03_L"]
ring_l = ["ring00_L", "ring01_L", "ring02_L", "ring03_L"]
pinky_l = ["pinky00_L", "pinky01_L", "pinky02_L", "pinky03_L"]
#right hand
thumb_r = ["thumb01_R", "thumb02_R", "thumb03_R"]
index_r = ["index00_R", "index01_R", "index02_R", "index03_R"]
middle_r = ["middle00_R", "middle01_R", "middle02_R", "middle03_R"]
ring_r = ["ring00_R", "ring01_R", "ring02_R", "ring03_R"]
pinky_r = ["pinky00_R", "pinky01_R", "pinky02_R", "pinky03_R"]

MB_fingers = thumb_l + index_l + middle_l + ring_l + pinky_l + thumb_r + index_r + middle_r + ring_r + pinky_r

############################################################################################################

MBx = ["pelvis", "pelvis", "upperarm_twist_L", "upperarm_twist_R", "lowerarm_twist_L", "lowerarm_twist_R", "thigh_twist_L", "thigh_twist_R", "calf_twist_L", "calf_twist_R", "neck"]
MRx = ["pelvis.L", "pelvis.R", "upper_arm.L.001", "upper_arm.R.001", "forearm.L.001", "forearm.R.001", "thigh.L.001", "thigh.R.001", "shin.L.001", "shin.R.001", "spine.004"]

MB_parts = MB_list + MB_fingers + MBx
MR_parts = MR_list + MR_fingers + MRx

############################################################################################################

spine = ["tweak_spine", "tweak_spine.001", "tweak_spine.002", "tweak_spine.003", "tweak_spine.004", "tweak_spine.005",]
torso = ["hips", "chest", "neck", "head"]
fk_arms = ["upper_arm_fk.L", "forearm_fk.L", "upper_arm_fk.R", "forearm_fk.R"]
fk_legs = ["thigh_fk.L", "shin_fk.L", "thigh_fk.R", "shin_fk.R"]
tw_armL = ['hand_tweak.L', 'forearm_tweak.L', 'upper_arm_tweak.L']
tw_armR = ['hand_tweak.R', 'forearm_tweak.R', 'upper_arm_tweak.R']
tw_legL = ['foot_tweak.L', 'shin_tweak.L', 'thigh_tweak.L']
tw_legR = ['foot_tweak.R', 'shin_tweak.R', 'thigh_tweak.R']
rig_spT = ["head", "neck", "tweak_spine.005"]
rig_spB = ["torso", "chest", "tweak_spine.005"]

aL = ["PUL_Hand.L", "PUL_Forearm.L", "PUL_Upperarm.L"]
aR = ["PUL_Hand.R", "PUL_Forearm.R", "PUL_Upperarm.R"]
lL = ["PUL_Foot.L", "PUL_Shin.L", "PUL_Thigh.L"]
lR = ["PUL_Foot.R", "PUL_Shin.R", "PUL_Thigh.R"]
spT = ["PUL_head", "PUL_neck", "PUL_CenterT"]
spB = ["PUL_torso", "PUL_chest", "PUL_CenterB"]

pull_links = [["PUL_Upperarm.L", "PUL_CenterB", "PUL_Upperarm.R"], ["PUL_Thigh.L", "PUL_torso", "PUL_Thigh.R"]]

blist = [tw_armL, tw_armR, tw_legL, tw_legR, rig_spT, rig_spB]
tlist = [aL, aR, lL, lR, spT, spB]

############################################################################################################



root_b = [28]
rig_bones = [3,5]
tweak_bones = [4,6,9,12,15,18]
ik_bones = [7,10,13,16]
fk_bones = [8,11,14,17]

def ragdoll_dict(part_names):
    rd = {
        part_names[0]: [-25, 45, -45, 45, -30, 30, part_names[1], .06],             #head  #x -22, 37,
        part_names[1]: [-25, 45, -45, 45, -30, 30, part_names[18], .02],            #neck  #x -22, 37,
        part_names[2]: [-30, 30, 0, 0, -30, 10, part_names[1], .05],                #clavicle_L
        part_names[3]: [-30, 30, 0, 0, -10, 30, part_names[1], .05],                #clavicle_R
        part_names[20]: [-10, 10, 0, 0, 0, 0, part_names[18], .05],                 #breast_L
        part_names[21]: [-10, 10, 0, 0, 0, 0, part_names[18], .05],                 #breast_R
        part_names[19]: [-22, 45, -45, 45, -15, 15, '', .1],                        #pelvis
        part_names[16]: [-55, 68, -45, 45, -30, 30, part_names[19], .1],            #spine_01
        part_names[17]: [-45, 45, -45, 45, -30, 30, part_names[16], .2],            #spine_02
        part_names[18]: [-45, 45, -45, 45, -30, 30, part_names[17], .1],            #spine_03
        part_names[4]: [-58, 95, -30, 15, -60, 105, part_names[2], .03],            #upperarm_L
        part_names[5]: [-58, 95, -30, 15, -60, 105, part_names[3], .03],            #upperarm_R
        part_names[6]: [-146, 0, -15, 0, 0, 0, part_names[4], .014],                #lowerarm_L
        part_names[7]: [-146, 0, -15, 0, 0, 0, part_names[5], .014],                #lowerarm_R
        part_names[8]: [-30, 30, -15, 15, -25, 36, part_names[6], .006],            #hand_L
        part_names[9]: [-30, 30, -15, 15, -36, 25, part_names[7], .006],            #hand_R
        part_names[10]: [-90, 45, -15, 15, -22, 17, part_names[19], .1],            #thigh_L
        part_names[11]: [-90, 45, -15, 15, -22, 17, part_names[19], .1],            #thigh_R
        part_names[12]: [0, 150, 0, 0, -3, 3, part_names[10], .05],                 #calf_L
        part_names[13]: [0, 150, 0, 0, -3, 3, part_names[11], .05],                 #calf_R
        part_names[14]: [-44, 45, -26, 26, -15, 74, part_names[12], .01],           #foot_L
        part_names[15]: [-45, 44, -26, 26, -74, 15, part_names[13], .01],           #foot_R
        }
    return rd

############################################################################################################

def timeit(method):
    import time
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((ts - te) * 1000)
        else:
            print("{}, {} milliseconds".format(method.__name__, ((te - ts) * 1000)))
        return result
    return timed

def get_os():
    import platform
    gos = platform.system()
    if gos == 'Windows':
        return 'win'
    if gos == 'Darwin':
        return 'mac'
    else:
        return 'linux'

def get_addons():
    paths_list = paths()
    addon_list = []
    for path in paths_list:
        for modName, modPath in bpy.path.module_names(path):
            is_enabled, is_loaded = check(modName)
            addon_list.append(modName)
    return(addon_list)

def enable_addon(dependencies):
    addons = get_addons()
    for addon in dependencies:
        is_enabled, is_loaded = check(addon)
        if not is_enabled:
            enable(addon, default_set=True)
        else:
            if addon in addons:
                print("Error {} is Already Enabled".format(addon))
            else:
                print("Error {} is Missing ".format(addon))

def name_check(Name):
    obj = bpy.data.objects
    onames = [i.name for i in obj]
    if Name not in onames:
        return Name
    else:
        return obj[Name].copy().name

def inc_trip_num(num):
    n = np.array([int(i) for i in str(num)])
    c1 = len(n)
    if (c1 > 3) or (int(num) == 999):
        print("Exceed trip num limit!")
        pass
    else:
        if c1 == 1:
            n[0] = n[0] + 1
            if n[0] == 10:
                return "010"
            else:
                return "00{}".format(n[0])
        if c1 == 2:
            n[1] = n[1] + 1
            if n[1] >= 10:
                n[0] = n[0] + 1
                if n[0] == 10:
                    return "100"
                else:
                    return "0{}0".format(n[0])
            else:
                return "0{}{}".format(n[0], n[1])
        if c1 == 3:
            n[2] = n[2] + 1
            if n[2] == 10:
                n[1] = n[1] + 1
                if n[1] == 10:
                    n[0] = n[0] + 1
                    return "{}00".format(n[0])
                else:
                    return "{}{}0".format(n[0], n[1])
            else:
                return "{}{}{}".format(n[0], n[1], n[2])

def name_verify(Name, collection):
    get_collection(collection)
    cl = collection_object_list(collection)
    search = [i for i in cl if Name in i]
    count = len(search)
    if count > 0:
        if count == 1:
            return "{}_001".format(Name)
        if count >= 2:
            try:
                nums = np.array([int(i[-3:]) for i in search if (i[-3:].isdigit()) and (len(i[:-4]) == len(Name))])
                num = nums.max()
                n = inc_trip_num(num)
                return "{}_{}".format(Name, n)
            except:
                pass
    else:
        return Name

def get_obj_collection(ob):
    coll_dict = {collection.name: [o.name for o in collection.all_objects] for collection in bpy.data.collections}
    coll = np.array(list(coll_dict.keys()))
    obs = np.array(list(coll_dict.values()))
    gc = coll[np.argwhere([ob in i for i in obs]).ravel()[0]]
    return gc

def collection_object_list(collection):
    return [o.name for o in bpy.data.collections[collection].objects[:]]

def new_collection(Name):
    new_coll = bpy.data.collections.new(Name)
    bpy.context.scene.collection.children.link(new_coll)
    return new_coll

def new_subcollection(Name, collection):
    new_coll = bpy.data.collections.new(Name)
    bpy.data.collections[collection].children.link(new_coll)
    return new_coll

def get_collection(Name):
    if bpy.data.collections.get(Name) == None:
        return new_collection(Name)
    else:
        return bpy.data.collections.get(Name)

def get_subcollection(Name, collection):
    if (bpy.data.collections.get(collection) == None) and (bpy.data.collections.get(Name) == None):
        new_collection(collection)
        return new_subcollection(Name, collection)
    elif bpy.data.collections.get(Name) == None:
        return new_subcollection(Name, collection)
    else:
        return bpy.data.collections.get(Name)

def list_find(List, cond):
    return np.array([i for i in List if cond in i])

def list_find_inv(List, cond):
    return np.array([i for i in List if cond not in i])

###############################################################################################################################
# OBJECT OPS

@timeit
def fibonacci(num):
    nm = num + 1
    n = np.arange(1, nm)
    s5 = np.sqrt(5)
    phi = (1 + s5)/2
    return np.rint((phi**n - (-1/phi)**n)/s5)

def fib_at_index(num):
    n = num - 1
    F = fibonacci(num)
    return F.tolist()[n]

def spiral_map(vec, count):
    vec = np.array(vec)
    vc = vec.copy()
    grad = np.arange(1, count)
    rm90 = rotation_matrix(0, 0, np.radians(90))
    fib = fib_at_index
    sm = [vec,]
    for i in grad:
        vc = vc + ((vc * fib(i)) @ rm90)
        sm.append(vc)
    return np.array(sm)

def rotation_matrix(xrot, yrot, zrot):
    #rotation_matrix(x rotation, y rotation, z rotation) #in radians
    rot_mat = np.array(
                    [ [np.cos(-zrot)*np.cos(yrot), -np.sin(-zrot)*np.cos(xrot) + np.cos(-zrot)*np.sin(yrot)*np.sin(xrot), np.sin(-zrot)*np.sin(xrot) + np.cos(-zrot)*np.sin(yrot)*np.cos(xrot)],
                    [ np.sin(-zrot)*np.cos(yrot), np.cos(-zrot)*np.cos(xrot) + np.sin(-zrot)*np.sin(yrot)*np.sin(xrot), -np.cos(-zrot)*np.sin(xrot) + np.sin(-zrot)*np.sin(yrot)*np.cos(xrot)],
                    [-np.sin(yrot), np.cos(yrot)*np.sin(xrot), np.cos(yrot)*np.cos(xrot)] ]
                    )
    return rot_mat

def edge_idx_chain(count):
    ch = np.arange(count)
    c1 = ch[:-1]
    c2 = ch[1:]
    return np.array(list(zip(c1, c2)))

def edge_idx_loop(count):
    ch = np.arange(count)
    ec = edge_idx_chain(count)
    return np.append(ec, [[ch[-1], ch[0]]], axis=0)

def hide_ob(ob, Bool):
    obj = bpy.data.objects
    o = obj[ob]
    o.hide_viewport = Bool
    o.hide_render = Bool

def active_ob(object, objects):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects[object].select_set(state=True)
    bpy.context.view_layer.objects.active = bpy.data.objects[object]
    if objects is not None:
        for o in objects:
            bpy.data.objects[o].select_set(state=True)

def obj_mesh(co, edges, faces, collection):
    cur = bpy.context.object
    mesh = bpy.data.meshes.new("Obj")
    mesh.from_pydata(co, edges, faces)
    mesh.validate()
    mesh.update(calc_edges=True) if edges == [] else None
    Object = bpy.data.objects.new("Obj", mesh)
    Object.data = mesh
    bpy.data.collections[collection].objects.link(Object)
    bpy.context.view_layer.objects.active = Object

@timeit
def obj_new(Name, co, edges, faces, collection):
    obj_mesh(co, edges, faces, collection)
    bpy.data.objects["Obj"].name = Name
    bpy.data.meshes[bpy.data.objects[Name].data.name].name = Name

def adoption(parent, child, type, index):
    '''types: OBJECT, ARMATURE, LATTICE, VERTEX, VERTEX_3, BONE'''
    par = bpy.data.objects[parent]
    ch = bpy.data.objects[child]
    ch.parent = par
    ch.matrix_world = par.matrix_world @ par.matrix_world.inverted()
    ch.parent_type = type
    if type == 'VERTEX':
        ch.parent_vertices[0] = index
    if type == 'BONE':
        ch.parent_bone = index

def add_parent(parent, children):
    active_ob(parent, children)
    bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
    bpy.ops.object.select_all(action='DESELECT')

# ------------------------------------------------------------------------
#    Copy_Object Class
# ------------------------------------------------------------------------
@timeit
class CopyObject:
    '''
    '''
    def __init__(self, ob, Name):
        self.name = Name
        self.ob = ob
        self.location = self.ob.location
        self.rotation = self.ob.rotation_euler
        self.matrix_world = self.ob.matrix_world
        self.matrix_inverted = self.ob.matrix_world.inverted()
        self.mesh = ob.data
        self.verts = ob.data.vertices
        self.edges = ob.data.edges
        self.faces = ob.data.polygons
        self.vert_groups = ob.vertex_groups
        self.shape_keys = ob.data.shape_keys
        self.vert_count = len(self.verts)
        self.vert_indexes = np.arange(self.vert_count, dtype=np.int32)
        self.edge_count = len(self.edges)
        self.edge_indexes = np.arange(self.edge_count, dtype=np.int32)
        self.face_count = len(self.faces)
        self.face_indexes = np.arange(self.face_count, dtype=np.int32)
        self.co = np.empty(self.vert_count * 3, dtype=np.float32)
        self.verts.foreach_get('co', self.co)
        self.is_selected_verts = np.empty(self.vert_count, dtype=np.bool)
        self.verts.foreach_get('select', self.is_selected_verts)
        self.selected_verts = self.vert_indexes[self.is_selected_verts]
        self.selected_co = np.array([self.verts[i].co for i in (j for j in self.selected_verts)])
        self.is_selected_edges = np.empty(self.edge_count, dtype=np.bool)
        self.edges.foreach_get('select', self.is_selected_edges)
        self.selected_edges = self.edge_indexes[self.is_selected_edges]
        self.edge_verts = np.array([i.vertices[:] for i in (j for j in self.edges)])
        self.is_selected_faces = np.empty(self.face_count, dtype=np.bool)
        self.faces.foreach_get('select', self.is_selected_faces)
        self.selected_faces = self.face_indexes[self.is_selected_faces]
        self.face_verts = np.array([i.vertices[:] for i in (j for j in self.faces)])
        self.new_vert_count = self.selected_verts.size
        self.new_edge_count = self.selected_edges.size
        self.new_face_count = self.selected_faces.size
        self.new_vert_indexes = np.arange(self.new_vert_count, dtype=np.int32)
        self.new_vert_dict = {o: n for n, o in enumerate(self.selected_verts.tolist())}
        self.new_edge_verts = [[self.new_vert_dict[i] for i in nest] for nest in (j for j in self.edge_verts[self.is_selected_edges])]
        self.new_face_verts = [[self.new_vert_dict[i] for i in nest] for nest in (j for j in self.face_verts[self.is_selected_faces])]
        self.kd_tree = self.get_kd_tree(self.ob)
    '''
    '''
    '''
    '''
    #vertex group index list
    def vg_idx_list(self, vgn):
        return([v.index for v in (g for g in self.verts) if v.select and self.vert_groups[vgn].index in [vg.group for vg in v.groups]])
    '''
    '''
    #vertex group {name: [indexes]} dictionary
    def vert_group_idx_dict(self):
        vn = [v.name for v in self.vert_groups[:]]
        vd = {n: self.vg_idx_list(n) for n in vn}
        vdd = {k: vd[k] for k in vd if vd[k] != []}
        return {d: [self.new_vert_dict[i] for i in vdd[d]] for d in (v for v in vdd)}
    '''
    '''
    #vertex group index list
    def vidx_list(self, vgn):
        return([[v.index, v.groups[0].weight] for v in self.verts if v.select and self.vert_groups[vgn].index in [vg.group for vg in v.groups]])
    '''
    '''
    #vertex group {name: [indexes]} dictionary vidx_dict
    def vert_idx_dict(self):
        vn = [v.name for v in self.vert_groups[:]]
        vd = {n: self.vidx_list(n) for n in (v for v in vn)}
        vdd = {k: vd[k] for k in (v for v in vd) if vd[k] != []}
        return vdd
    '''
    '''
    def rotation_matrix(self, xrot, yrot, zrot):
        #rotation_matrix(x rotation, y rotation, z rotation) #in radians
        rot_mat = np.array(
                        [ [np.cos(-zrot)*np.cos(yrot), -np.sin(-zrot)*np.cos(xrot) + np.cos(-zrot)*np.sin(yrot)*np.sin(xrot), np.sin(-zrot)*np.sin(xrot) + np.cos(-zrot)*np.sin(yrot)*np.cos(xrot)],
                        [ np.sin(-zrot)*np.cos(yrot), np.cos(-zrot)*np.cos(xrot) + np.sin(-zrot)*np.sin(yrot)*np.sin(xrot), -np.cos(-zrot)*np.sin(xrot) + np.sin(-zrot)*np.sin(yrot)*np.cos(xrot)],
                        [-np.sin(yrot), np.cos(yrot)*np.sin(xrot), np.cos(yrot)*np.cos(xrot)] ]
                        )
        return rot_mat
    '''
    '''
    def new_object(self):
        cur = self.ob
        mesh = bpy.data.meshes.new(self.name)
        rot = (self.rotation.x, self.rotation.y, self.rotation.z)
        co = self.selected_co
        mesh.from_pydata(co, [], self.new_face_verts)
        mesh.validate()
        mesh.update(calc_edges = True)
        Object = bpy.data.objects.new(self.name, mesh)
        Object.data = mesh
        bpy.context.collection.objects.link(Object)
        bpy.context.view_layer.objects.active = Object
        Object.matrix_world = Object.matrix_world.copy() @ self.matrix_world
        cur.select_set(False)
        Object.select_set(True)
    '''
    '''
    def new_ob_w_weights(self):
        self.new_object()
    '''
    '''
    def new_object_full(self):
        self.new_ob_w_weights()
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN')
        bpy.ops.object.shade_smooth()
    '''
    '''
    #transfer vertex weight to new object
    def transfer_vt(self):
        viw = self.vert_group_idx_dict()
        vg = bpy.context.object.vertex_groups
        vt = self.verts
        for vgroup in viw:
            nvg = bpy.context.object.vertex_groups.new(name=vgroup)
            nvg.add(viw[vgroup], 1.0, "ADD")
    '''
    '''
    def add_wt(self):
        vid = self.vert_idx_dict()
        vt = self.verts
        for v in vid:
            for i in vid[v]:
                vt[i[0]].groups[0].weight = i[1]
    '''
    '''
    def copy_wt(self):
        self.transfer_vt()
        self.add_wt()
    '''
    '''
    def centroid(self, co):
        co = np.array(co, float)
        return co.mean(axis=0)
    '''
    '''
    def vert_edges(self, idx):
        vt = self.verts
        vc = self.vert_count
        ed = self.edges
        ec = self.edge_count
        ev = np.empty((ec * 2),int)
        ed.foreach_get('vertices', ev)
        ev.shape = (ec, 2)
        mask = lambda i: ev[np.any(np.isin(ev, i), axis=-1)]
        ve = mask(idx)
        return ve
    '''
    '''
    def vert_edge_map(self):
        vt = self.verts
        vc = self.vert_count
        ed = self.edges
        ec = self.edge_count
        ev = np.empty((ec * 2),int)
        ed.foreach_get('vertices', ev)
        ev.shape = (ec, 2)
        mask = lambda li: np.array([ev[np.any(np.isin(ev, j), axis=-1)] for j in range(len(li))])
        vem = np.empty(vc*2,)
        vem = mask(vt)
        return vem
    '''
    '''
    def vert_edge_vec(self, idx):
        vt = self.verts
        ve = vert_edges(self, idx)
        vec = lambda i: np.array(vt[i[1]].co) - np.array(vt[i[0]].co)
        vev = np.array([vec(i) for i in ve], float)
        return vev
    '''
    '''
    def new_material(self, Name, info, links):
        material = get_material(Name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        clear_node(material)
        for i in info:
            add_shader_node(material, i[0], i[2], i[3], i[4])
        for i in links:
            add_node_link(material, eval(i[0]), eval(i[1]))
    '''
    '''
    def set_material(self, material, style):
        nodes = material.node_tree.nodes
        args = style
        for arg in args:
            id = nodes[arg[0]].bl_idname
            fd = func_dict(shader_set_dict(), id)
            data = args[args.index(arg)]
            fd(*data)
    '''
    '''
    def get_kd_tree(self, object):
        vt = object.data.vertices
        count = len(vt)
        kd = kdtree.KDTree(count)
        for i, v in enumerate(vt):
            kd.insert(v.co, i)
        kd.balance()
        return kd
    '''
    '''
    def get_lerp(self, first, second, factor):
        first = Vector(first)
        second = Vector(second)
        l = first.lerp(second, factor)
        return np.array(l, float)
    '''
    '''
    def get_slerp(self, first, second, factor):
        first = Vector(first)
        second = Vector(second)
        s = first.slerp(second, factor)
        return np.array(s, float)
    '''
    '''
    def get_mesh(self):
        if bpy.context.object.type == 'ARMATURE':
            return bpy.context.object.children[0]
        else:
            return bpy.context.object
    '''
    '''
    def get_skeleton(self):
        if bpy.context.object.type == 'ARMATURE':
            return bpy.context.object
        else:
            return bpy.context.object.parent
    '''
    '''
    def transfer_shape_keys(self, other):
        tr = Transfer(self.ob)
        tr.transfer(other)


###############################################################################################################################
# VERT GROUP OPS

def add_vert_group(ob, vgroup, index):
    nvg = bpy.data.objects[ob].vertex_groups.new(name=vgroup)
    nvg.add(index, 1.0, "REPLACE")

@timeit
def hair_vg(ob, target):
    obj = bpy.data.objects
    o = obj[ob]
    vt = o.data.vertices
    vg = o.vertex_groups
    cv = len(vt)
    tl = obj[target].location
    co = np.empty(cv * 3)
    vt.foreach_get("co", co)
    co = co.reshape(cv, 3) + np.array(o.location)
    # indexes
    idx_top = np.argwhere(co[:,2] >= tl.z).ravel()
    idx_bottom = np.argwhere(co[:,2] <= tl.z).ravel()
    idx_left = np.argwhere(co[:,0] >= tl.x).ravel()
    idx_right = np.argwhere(co[:,0] <= tl.x).ravel()
    idx_front = np.argwhere(co[:,1] >= tl.y).ravel()
    idx_back = np.argwhere(co[:,1] <= tl.y).ravel()
    idx_lt = np.intersect1d(idx_top, idx_left)
    idx_rt = np.intersect1d(idx_top, idx_right)
    idx_lb = np.intersect1d(idx_bottom, idx_left)
    idx_rb = np.intersect1d(idx_bottom, idx_right)
    # vert groups
    new_vgs = ["top", "bottom", "left", "right", "front", "back", "top_left", "top_right", "bottom_left", "bottom_right"]
    new_idxs = [idx_top, idx_bottom, idx_left, idx_right, idx_front, idx_back, idx_lt, idx_rt, idx_lb, idx_rb]
    for i, v in list(zip(new_idxs, new_vgs)):
        add_vert_group(ob, v, i.tolist())

def hair_splitter_target(ob):
    obj = bpy.data.objects
    add_emp = lambda l: bpy.ops.object.empty_add(type='PLAIN_AXES', radius=1.0, align='WORLD', location=l, rotation=(0,0,0))
    add_emp(obj[ob].location)
    bpy.context.object.name = "Hair_Splitter"

@timeit
def add_hair_vg(ob):
    obj = bpy.data.objects
    target = "Hair_Splitter"
    hair_vg(ob, target)
    active_ob(target, None)
    bpy.ops.object.delete(use_global=False, confirm=False)

@timeit
def get_vert_groups_wts(Obj):
    obj = bpy.data.objects
    body = obj[Obj]
    vg = body.vertex_groups
    vt = body.data.vertices
    vn = [v.name for v in vg]
    vgi = [[[vg.group] for vg in v.groups] for v in vt]
    vgw = [[[vg.weight] for vg in v.groups] for v in vt]
    return vn, vgi, vgw

@timeit
def set_vert_groups_wts(Obj, v_names, indexes, weights):
    obj = bpy.data.objects
    body = obj[Obj]
    vg = body.vertex_groups
    vt = body.data.vertices
    count = len(vt)
    vi = np.arange(count)
    gw = lambda i, g: np.isin(g, [i])
    for i, n in enumerate(v_names):
        nvg = vg.new(name=n)
        mask = np.array([gw(i, w).any() for w in indexes])
        index = vi[mask].tolist()
        nvg.add(index, 1.0, 'REPLACE')
    for idx, v in enumerate(weights):
        for ix, w in enumerate(v):
            vt[idx].groups[ix].weight = w


###############################################################################################################################
# MODIFIER OPS

#Add modifier
def add_modifier(Object, Name, Type):
    Object.modifiers.new(Name, type=Type)

#Apply Modifier
def apply_mod(Ref):
    act = bpy.context.view_layer.objects.active
    for o in bpy.context.view_layer.objects:
        for m in o.modifiers:
            if Ref in m.name:
                bpy.context.view_layer.objects.active = o
                bpy.ops.object.modifier_apply(modifier=m.name)
    bpy.context.view_layer.objects.active = act

def add_cloth(object, v_group):
    add_modifier(object, 'Cloth', 'CLOTH')
    cm = Object.modifiers['Cloth']
    cm.settings.vertex_group_mass = v_group
    cm.settings.use_dynamic_mesh = True

def add_shrinkwrap(Object, Target):
    add_modifier(Object, 'shrink', 'SHRINKWRAP')
    oms = Object.modifiers['shrink']
    oms.target = Target

###############################################################################################################################

def bone_collector(armature, Dict=True):
    bones = bpy.data.armatures[armature].bones
    count = len(bones)
    bc = np.empty((count, 2))
    bn = np.empty(count, dtype='U10')
    bhd = np.empty((count * 3), dtype=float).round(decimals=5)
    bhr = np.empty(count, dtype=float).round(decimals=5)
    btl = np.empty((count * 3), dtype=float).round(decimals=5)
    btr = np.empty(count, dtype=float).round(decimals=5)
    bl = np.empty(count, dtype=float).round(decimals=5)
    bn = np.asarray([i.name for i in bones])
    bones.foreach_get('head_local', bhd)
    bones.foreach_get('head_radius', bhr)
    bones.foreach_get('tail_local', btl)
    bones.foreach_get('tail_radius', btr)
    bones.foreach_get('length', bl)
    bhd.shape = ((count, 3))
    btl.shape = ((count, 3))
    bc = list(zip(bn, np.array(list(zip(bhd, btl, bhr, btr, bl)))))
    if Dict == False:
        return np.array(bc)
    else:
        return dict(bc)

def bone_matrix(armature, Dict=True):
    bones = bpy.data.armatures[armature].bones
    count = len(bones)
    bc = np.empty((count * 16), dtype=float)
    bn = np.asarray([i.name for i in bones])
    bones.foreach_get('matrix', bc)
    bc.shape = ((count, 4, 4))
    bc = list(zip(bn, np.array(list(zip(bc)))))
    if Dict == False:
        return np.array(bc)
    else:
        return dict(bc)

def add_armature(Name, collection, type='STICK'):
    ''' 'OCTAHEDRAL', 'STICK', 'BBONE', 'ENVELOPE', 'WIRE' '''
    arm = bpy.data.armatures.new(Name)
    obj = bpy.data.objects
    onew = obj.new(Name, arm)
    onew.data.display_type = type
    bpy.data.collections[collection].objects.link(onew)

def add_bone(armature, Name, co):
    obj = bpy.data.objects
    arm = obj[armature]
    eb = arm.data.edit_bones
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
    b = eb.new(Name)
    b.head = co[0]
    b.tail = co[1]
    b.use_connect = True
    bpy.ops.object.mode_set(mode='OBJECT')

def hair_bones(Name):
    coll = bpy.context.collection
    cl = bpy.data.collections
    v_group = "pin"
    hcoll = 'Hair_Accessories'
    tcoll = 'Hair_Targets'
    ccoll = 'Hair_Controls'
    a_name = "ARM_{}".format(Name)
    get_collection(hcoll)
    get_subcollection(tcoll, hcoll)
    get_subcollection(ccoll, hcoll)
    obj = bpy.data.objects
    curve = obj[Name]
    add_emp = lambda l, r: bpy.ops.object.empty_add(type='CUBE', radius=0.01, align='WORLD', location=l, rotation=r)
    add_edge = lambda Name, co, edges: obj_new(Name, co, edges, [], ccoll)
    active_ob(Name, None)
    bpy.ops.object.convert(target='MESH', keep_original=True)
    me = bpy.context.object
    e_name = "EDG_{}".format(Name)
    me.name = e_name
    cl[ccoll].objects.link(me)
    coll.objects.unlink(me)
    vt = me.data.vertices
    count = len(vt)
    co = np.empty(count * 3)
    vt.foreach_get('co', co)
    co.shape = (count, 3)
    mw = np.array(curve.matrix_world.inverted().to_3x3())
    co = (co @ mw) + np.array(curve.location)
    count = len(co)
    edges = None
    edges = edge_idx_chain(count)
    add_armature(a_name, hcoll)
    bones = []
    for i, v in enumerate(edges):
        n = "{}_B{}".format(a_name, i)
        add_bone(a_name, n, co[v])
        bones.append(n)
    bones.reverse()
    arm = obj[a_name]
    bpy.context.view_layer.objects.active = arm
    pb = arm.pose.bones
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
    eb = arm.data.edit_bones
    for idx, b in enumerate(bones[:-1]):
        ix = idx + 1
        eb[b].parent = eb[bones[ix]]
        eb[b].use_connect = True
    bpy.ops.object.mode_set(mode='OBJECT')
    targets = []
    bones.reverse()
    for i, v in enumerate(co[:-1]):
        nm = "TAR_{}_{}".format(a_name, i)
        r = pb[bones[i]].matrix.to_euler('XYZ')
        add_emp(v, r)
        obj["Empty"].name = nm
        cl[tcoll].objects.link(obj[nm])
        coll.objects.unlink(obj[nm])
        targets.append(nm)
    tidx = count - 1
    nml = "TAR_{}_{}".format(a_name, tidx)
    rl = pb[bones[-1]].matrix.to_euler('XYZ')
    add_emp(co[tidx], rl)
    obj["Empty"].name = nml
    cl[tcoll].objects.link(obj[nml])
    coll.objects.unlink(obj[nml])
    targets.append(nml)
    for ix, k in enumerate(targets):
        active_ob(e_name, [k])
        bpy.ops.object.mode_set(mode='EDIT', toggle=False)
        par = bpy.data.objects[e_name]
        ch = bpy.data.objects[k]
        ch.parent = par
        ch.matrix_world = par.matrix_world
        ch.parent_type = 'VERTEX'
        ch.parent_vertices[0] = ix
        bpy.ops.object.mode_set(mode='OBJECT')
    active_ob(a_name, None)
    for b, t in list(zip(bones, targets[1:])):
        bct = pb[b].constraints.new('DAMPED_TRACK')
        bct.target = obj[t]
    bpy.ops.object.select_all(action='DESELECT')


#---------------------------------------------------------

def PK_limit_distance(Obj, tar, lm, dist, Name):
    obj = bpy.data.objects
    ob = obj[Obj]
    ld = ob.constraints.new("LIMIT_DISTANCE")
    ld.target = obj[tar]
    ld.limit_mode = lm
    ld.distance = dist
    ld.name = Name

@timeit
def capsule_PK(Bone, target, Scale):
    db = bone_collector(0, Dict=True)[Bone]
    edge_maker = lambda Name, co: obj_new(Name, co, [[0, 1]], [], "PK_Ragdoll")
    Name = "CAP_{}".format(Bone)
    co = db[[0, 1]]
    rad = (db[2] * Scale)
    edge_maker(Name, co)
    add_modifier(bpy.data.objects[Name], "Capsule", 'SKIN')
    add_modifier(bpy.data.objects[Name], "SubSurf", 'SUBSURF')
    bpy.data.objects[Name].modifiers["Capsule"].use_smooth_shade = True
    bpy.data.objects[Name].data.skin_vertices[""].data[0].radius[:] = (rad, rad)
    bpy.data.objects[Name].data.skin_vertices[""].data[1].radius[:] = (rad, rad)

@timeit
def ragdoll_assemble(List, target, Scale):
    #new_collection("PK_Ragdoll")
    for bone in List:
        capsule_PK(bone, target, Scale)

@timeit
def show_rig(object, idx, Bool):
    for r in idx:
        object.data.layers[r] = Bool

@timeit
def PK_limb_build():
    coll = bpy.context.collection
    pkcoll = get_collection("Pino_Kio_Pull")
    pkrag = get_collection("PK_Ragdoll")
    obj = bpy.data.objects
    bc = bone_collector("rig")
    li = 'LIMITDIST_INSIDE'
    lo = 'LIMITDIST_OUTSIDE'
    ld = PK_limit_distance
    add_emp = lambda l: bpy.ops.object.empty_add(type="SPHERE", radius=0.1, align='WORLD', location=l, rotation=(0,0,0))
    edge_maker = lambda Name, co: obj_new(Name, co, [[0, 1]], [], pkrag.name)
    get_ll = lambda List: np.array([np.array([bc[b][0], bc[b][1], bc[b][2]]) for b in List])
    loclen = np.array([get_ll(L) for L in blist])
    edges = []
    for ib, b in enumerate(tlist):
        t1 = []
        for ii, i in enumerate(b):
            h = loclen[ib][ii][0]
            add_emp(h)
            obj["Empty"].name = i
            pkcoll.objects.link(obj[i])
            coll.objects.unlink(obj[i])
            if i != b[-1]:
                e1 = "{}_Ragdoll".format(b[(ii + 1)])
                rad = loclen[ib][ii][2]
                t1.append(e1)
                t = loclen[ib][(ii + 1)][0]
                edge_maker(e1, [h,t])
                add_modifier(obj[e1], "Capsule", 'SKIN')
                add_modifier(obj[e1], "SubSurf", 'SUBSURF')
                obj[e1].modifiers["Capsule"].use_smooth_shade = True
                obj[e1].data.skin_vertices[""].data[0].radius[:] = (rad, rad)
                obj[e1].data.skin_vertices[""].data[1].radius[:] = (rad, rad)
            else:
                pass
        edges.append(np.array(t1))
    edges = np.array(edges)
    #############################
    for ix, x in enumerate(tlist):
        ln = loclen[ix]
        l1 = np.linalg.norm((ln[1][0] - ln[0][0]))
        l2 = np.linalg.norm((ln[2][0] - ln[1][0]))
        l3 = l1 + l2
        o1 = x[0]
        o2 = x[1]
        o3 = x[2]
        ld(o1, o2, li, l1, "LD_1I2")
        ld(o1, o3, li, l3, "LD_1I3")
        ld(o1, o2, lo, l1, "LD_1O2")
        ld(o2, o3, li, l2, "LD_2I3")
        ld(o2, o3, lo, l2, "LD_2O3")
        ld(o2, o1, lo, l1, "LD_2O1")
        ld(o2, o1, li, l1, "LD_2I1")
        ld(o3, o2, li, l2, "LD_3I2")
        ld(o3, o1, li, l3, "LD_3I1")
        ld(o3, o2, lo, l2, "LD_3O2")
    for i, j in enumerate(edges):
        ed1 = j[0]
        ed2 = j[1]
        lock_scale(ed1, [0,1,2], True)
        lock_scale(ed2, [0,1,2], True)
    for ix, x in enumerate(pull_links):
        o1 = x[0]
        o2 = x[1]
        o3 = x[2]
        l1 = np.linalg.norm((np.array(obj[o2].location) - np.array(obj[o1].location)))
        l2 = np.linalg.norm((np.array(obj[o3].location) - np.array(obj[o2].location)))
        l3 = l1 + l2
        ld(o1, o2, li, l1, "LD_LIC")
        ld(o1, o3, li, l3, "LD_LIR")
        ld(o1, o2, lo, l1, "LD_LOC")
        ld(o1, o3, lo, l3, "LD_LOR")
        ld(o2, o3, li, l2, "LD_CIR")
        ld(o2, o3, lo, l2, "LD_COR")
        ld(o2, o1, lo, l1, "LD_COL")
        ld(o2, o1, li, l1, "LD_CIL")
        ld(o3, o2, li, l2, "LD_RIC")
        ld(o3, o1, li, l3, "LD_RIL")
        ld(o3, o2, lo, l2, "LD_ROC")
        ld(o3, o1, lo, l3, "LD_ROL")

#---------------------------------------------------------

def bone_const_mute(Bool, constraint, bones):
    obj = bpy.data.objects
    pb = obj["rig"].pose.bones
    for bone in bones:
        pb[bone].constraints[constraint].mute = Bool

def set_rig_param(bone, param, val):
    '''
    ex:
    pb["thigh_parent.L"]["IK_Stretch"] = 0 #float
    pb["thigh_parent.L"]["pole_vector"] = 1 #int
    '''
    obj = bpy.data.objects
    pb = obj['rig'].pose.bones
    pb[bone][param] = val

def lock_scale(Obj, xyz, Bool, Pose=''):
    '''xyz = [0, 1, 2]'''
    obj = bpy.data.objects
    ob = obj[Obj]
    if Pose != '':
        ls = ob.pose.bones[Pose]
    else:
        ls = ob
    for i in xyz:
        ls.lock_scale[i] = Bool

def lock_rotation(Obj, xyz, Bool, Pose=''):
    '''xyz = [0, 1, 2]'''
    obj = bpy.data.objects
    ob = obj[Obj]
    if Pose != '':
        ls = ob.pose.bones[Pose]
    else:
        ls = ob
    for i in xyz:
        ls.lock_rotation[i] = Bool

def lock_location(Obj, xyz, Bool, Pose=''):
    '''xyz = [0, 1, 2]'''
    obj = bpy.data.objects
    ob = obj[Obj]
    if Pose != '':
        ls = ob.pose.bones[Pose]
    else:
        ls = ob
    for i in xyz:
        ls.lock_location[i] = Bool

def lock_meta_scale():
    bc = bone_collector('metarig')
    bones = list(bc.keys())
    for bone in bones:
        lock_scale('metarig', [0, 1, 2], True, bone)

def lock_rig_loc():
    bones = ["upper_arm_ik.L", "upper_arm_ik.R", "thigh_ik.L", "thigh_ik.R"]
    for bone in bones:
        lock_location('rig', [0, 1, 2], True, bone)
        bpy.ops.object.posemode_toggle()
        set_rig_param(bone, "pole_vector", 1)
        bpy.ops.object.posemode_toggle()

def rig_stretch_locks():
    bones = ["upper_arm_parent.L", "upper_arm_parent.R", "thigh_parent.L", "thigh_parent.R"]
    for bone in bones:
        set_rig_param(bone, "IK_Stretch", 0)

def lock_pk_rig_loc():
    obs = ["head_control", "chest_control"]
    for ob in obs:
        lock_location(ob, [0, 1, 2], True)

#---------------------------------------------------------


def get_bez_curve_pts(Name):
    active_ob(Name, None)
    bpy.ops.object.convert(target='MESH', keep_original=True)
    me = bpy.context.object
    vt = me.data.vertices
    count = len(vt)
    co = np.empty(count * 3)
    vt.foreach_get('co', co)
    co.shape = (count, 3)
    bpy.ops.object.delete()
    return co

def curve_to_bone(Name, collection, seq='chain'):
    a_name = "ARM_{}".format(Name)
    obj = bpy.data.objects
    curve = obj[Name]
    co = get_bez_curve_pts(Name)
    mw = np.array(curve.matrix_world.inverted().to_3x3())
    co = (co @ mw) + np.array(curve.location)
    count = len(co)
    edges = None
    if seq == 'chain':
        edges = edge_idx_chain(count)
    if seq == 'loop':
        edges = edge_idx_loop(count)
    else:
        print("seq must be 'chain' or 'loop'.")
    add_armature(a_name, collection, 'OCTAHEDRAL')
    bones = []
    for i, v in enumerate(edges):
        n = "{}_B{}".format(a_name, i)
        add_bone(a_name, n, co[v])
        bones.append(n)
    bones.reverse()
    arm = obj[a_name]
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
    eb = arm.data.edit_bones
    for idx, b in enumerate(bones[:-1]):
        ix = idx + 1
        eb[b].parent = eb[bones[ix]]
        eb[b].use_connect = True
    bpy.ops.object.mode_set(mode='OBJECT')

def get_all_curve_points(ob):
    obj = bpy.data.objects
    sp = obj[ob].data.splines
    count = len(sp)
    co = []
    for j in sp:
        c1 = len(j.points)
        a1 = np.empty(c1 * 4)
        j.points.foreach_get('co', a1)
        a1.shape = (c1, 4)
        b1 = a1[:,[0,1,2]]
        co.append(b1)
    return np.array(co)

def get_curve_pt(object):
    if object.type == 'CURVE':
        me = object.data
        sp = me.splines[0]
        pt = sp.points
        count = len(pt)
        co = np.empty(count * 4)
        pt.foreach_get('co', co)
        co.shape = (count, 4)
        return co[:,[0,1,2]]

def get_curve_pts(object, idx):
    if object.type == 'CURVE':
        me = object.data
        sp = me.splines[idx]
        bz = sp.bezier_points
        return [np.array([np.array(p.co), np.array(p.handle_left), np.array(p.handle_right)]) for p in bz]

def get_curve_4pts(List):
    return [[List[i][0], List[i][2], List[(i+1)][1], List[(i+1)][0]] for i in range((len(List) - 1))]

def slerp_cu_bez_curve(p1, h1, h2, p2, t):
    p1 = np.array(p1)
    p2 = np.array(p2)
    h1 = np.array(h1)
    h2 = np.array(h2)
    dist = np.linalg.norm((p2 - p1))
    per = (1 - t)
    p1 = p1 * (per**3)
    h1 = h1 * (3 * t * per**2)
    h2 = h2 * 3 * t**2 * per
    p2 = p2 * t**3
    return p1 + h1 + h2 + p2

def new_curve(Name, pts):
    bpy.ops.curve.primitive_bezier_curve_add(location=(0.0, 0.0, 0.0))
    cc = CopyCurve(bpy.context.object, Name)
    cc.set_curve_pts(0, pts)
    bpy.context.object.name = Name

class CopyCurve:
    '''
    '''
    def __init__(self, object, Name):
        self.curve = object
        self.name = Name
        self.data = self.curve.data
        self.splines = self.data.splines
        self.points = lambda i: self.get_curve_pts(i)
        self.points4 = lambda i: get_curve_4pts(self.points(i))
    '''
    '''
    def get_curve_pts(self, idx):
        sp = self.splines[idx]
        bz = sp.bezier_points
        return [np.array([np.array(p.co), np.array(p.handle_left), np.array(p.handle_right)]) for p in bz]
    '''
    '''
    def get_x_mirr_curve_pts(self, idx):
        co = np.array(self.get_curve_pts(idx))
        for p in co:
            p[:,0] = -p[:,0]
        return co
    '''
    '''
    def set_curve_pts(self, idx, List):
        sp = self.splines[idx]
        bz = sp.bezier_points
        for i, p in enumerate(List):
            bz[i].co = p[0]
            bz[i].handle_left = p[1]
            bz[i].handle_right = p[2]
    '''
    '''
    def get_curve_4pts(self, List):
        return [[List[i][0], List[i][2], List[(i+1)][1], List[(i+1)][0]] for i in range((len(List) - 1))]
    '''
    '''
    def slerp_cu_bez_curve(self, List, t):
        p1 = np.array(List[0])
        p2 = np.array(List[3])
        h1 = np.array(List[1])
        h2 = np.array(List[2])
        dist = np.linalg.norm((p2 - p1))
        per = (1 - t)
        p1 = p1 * (per**3)
        h1 = h1 * (3 * t * per**2)
        h2 = h2 * 3 * t**2 * per
        p2 = p2 * t**3
        return p1 + h1 + h2 + p2

#---------------------------------------------------------

def limit_bone_distance(Bone, tar, mode):
    obj = bpy.data.objects
    bcrig = bone_collector(-1)
    dist = np.linalg.norm((bcrig[Bone] - bcrig[tar])).tolist()[0]
    pb1 = obj["rig"].pose.bones.get(Bone)
    pb2 = obj["rig"].pose.bones.get(tar)
    bc = pb1.constraints.new("LIMIT_DISTANCE")
    bc.owner_space = 'LOCAL'
    bc.target = obj["rig"]
    bc.subtarget = tar
    bc.limit_mode = mode
    bc.distance = dist
    bc.owner_space = 'POSE'
    bc.target_space = 'POSE'

def limit_distance(Obj, tar, lm):
    obj = bpy.data.objects
    ob = obj[Obj]
    ld = ob.constraints.new("LIMIT_DISTANCE")
    ld.owner_space = 'LOCAL'
    ld.target = obj[tar]
    ld.limit_mode = lm

def bone_copy_transform(Obj, Bone, tar):
    obj = bpy.data.objects
    pb = obj[Obj].pose.bones.get(Bone)
    bc = pb.constraints.new("COPY_TRANSFORMS")
    bc.target = obj[tar]

def obj_copy_transform(Obj, tar):
    obj = bpy.data.objects
    ob = obj[Obj]
    ct = ob.constraints.new("COPY_TRANSFORMS")
    ct.target = obj[tar]

def copy_transform(Obj, tar, sub, ht):
    obj = bpy.data.objects
    ob = obj[Obj]
    ct = ob.constraints.new("COPY_TRANSFORMS")
    ct.target = obj[tar]
    ct.subtarget = sub
    ct.target_space = 'POSE'
    ct.head_tail = ht

def bone_copy_rot(Obj, Bone, tar, influence=1):
    obj = bpy.data.objects
    pb = obj[Obj].pose.bones.get(Bone)
    bc = pb.constraints.new("COPY_ROTATION")
    bc.target = obj[tar]
    bc.influence = influence

def copy_loc(Obj, tar):
    obj = bpy.data.objects
    cl = obj[Obj].constraints.new("COPY_LOCATION")
    cl.target = obj[tar]

def obj_limit_rot(Obj, limits, space):
    obj = bpy.data.objects
    lr = obj[Obj].constraints.new('LIMIT_ROTATION')
    lr.owner_space = space
    lr.use_limit_x = True
    lr.use_limit_y = True
    lr.use_limit_z = True
    lr.min_x = np.radians(limits[0])
    lr.max_x = np.radians(limits[1])
    lr.min_y = np.radians(limits[2])
    lr.max_y = np.radians(limits[3])
    lr.min_z = np.radians(limits[4])
    lr.max_z = np.radians(limits[5])

def bone_limit_rot(Bone, limits, space):
    obj = bpy.data.objects
    pb = obj["rig"].pose.bones
    lr = pb[Bone].constraints.new('LIMIT_ROTATION')
    lr.owner_space = space
    lr.use_limit_x = True
    lr.use_limit_y = True
    lr.use_limit_z = True
    lr.min_x = np.radians(limits[0])
    lr.max_x = np.radians(limits[1])
    lr.min_y = np.radians(limits[2])
    lr.max_y = np.radians(limits[3])
    lr.min_z = np.radians(limits[4])
    lr.max_z = np.radians(limits[5])

def bone_damp_trac(Bone, tar, track):
    obj = bpy.data.objects
    pb = obj["rig"].pose.bones
    dt = pb[Bone].constraints.new('DAMPED_TRACK')
    dt.target = obj[tar]
    dt.track_axis = track

def damp_trac(Obj, tar):
    obj = bpy.data.objects
    dt = obj[Obj].constraints.new('DAMPED_TRACK')
    dt.target = obj[tar]

def follow_path(object, tar):
    obj = bpy.data.objects
    fp = obj[object].constraints.new('FOLLOW_PATH')
    fp.target = obj[tar]
    fp.use_curve_radius = True

def head_track(Obj, tar):
    obj = bpy.data.objects
    ht = obj[Obj].constraints.new('DAMPED_TRACK')
    ht.target = obj[tar]
    ht.track_axis = 'TRACK_Z'

#---------------------------------------------------------

@timeit
def anim_path(Obj, path):
    obj = bpy.data.objects
    override = {'constraint': obj[Obj].constraints["Follow Path"]}
    bpy.ops.constraint.followpath_path_animate(override, constraint="Follow Path", owner='OBJECT')

@timeit
def path_control():
    add_emp = lambda t, rad, l, r: bpy.ops.object.empty_add(type=t, radius=rad, align='WORLD', location=l, rotation=r)
    obj = bpy.data.objects
    head_loc = obj["head_control"].location
    add_emp('SINGLE_ARROW', 0.1, (0, -1, head_loc[2]), (0, 0, 0))
    obj["Empty"].name = 'CTR_head'
    head_track("head_control", 'CTR_head')
    add_emp('SPHERE', 0.05, obj['hand_ik.L'].location, (0, 0, 0))
    obj["Empty"].name = 'CTR_hand.L'
    add_emp('SPHERE', 0.05, obj['hand_ik.R'].location, (0, 0, 0))
    obj["Empty"].name = 'CTR_hand.R'
    new_curve("PTH_hand.R", [[[0, 0, 0], [0.01472223, 0.70695353, 0], [-0.01472223, -0.70695353, 0]], [[0.5, -0.5, 0.5], [0.25, -0.5,  0.5], [1.5, -0.5, 0.5]]])
    new_curve("PTH_hand.L", [[[0, 0, 0], [-0.01472223, 0.70695353, 0], [0.01472223, -0.70695353, 0]], [[-0.5, -0.5, 0.5], [-0.25, -0.5,  0.5], [-1.5, -0.5, 0.5]]])
    copy_loc('hand_ik.L', 'CTR_hand.L')
    copy_loc('hand_ik.R', 'CTR_hand.R')
    follow_path('CTR_hand.L', 'PTH_hand.L')
    follow_path('CTR_hand.R', 'PTH_hand.R')
    anim_path('CTR_hand.L', 'PTH_hand.L')
    anim_path('CTR_hand.R', 'PTH_hand.R')

#---------------------------------------------------------

#######################################################################

def get_filename(filePath, File):
    FP = os.path.join(filePath, File)
    return FP

def get_hair_dir():
    ga = get_addons()
    addon = [i for i in ga if 'self_dot_accessories' in i][0]
    a_dir = bpy.utils.user_resource('SCRIPTS', "addons")
    return os.path.join(a_dir, addon)

def get_hair_npz(fileName):
    hair_dir = get_hair_dir()
    return os.path.join(hair_dir, fileName)

def get_universal_dict(filename):
    with np.load(filename, 'r+', allow_pickle=True) as data:
        d = dict(data)
    return d

def get_np(filename):
    with np.load(filename, 'r', allow_pickle=True) as data:
        d = {i: data[i] for i in data.files}
    return d

def get_universal_list(filename):
    with np.load(filename, 'r', allow_pickle=True) as data:
        l = list(data.files)
    return l

def get_universal_presets(filename, style):
    with np.load(filename, 'r', allow_pickle=True) as data:
        d = data[style]
    return d

def set_universal_shader(mat_name, filename, style):
    material = get_material(mat_name)
    nodes = material.node_tree.nodes
    ds = get_universal_presets(filename, style)
    args = [tuple(np.array(i).tolist()) for i in ds]
    for arg in args:
        id = nodes[arg[0]].bl_idname
        fd = func_dict(shader_set_dict(), id)
        data = args[args.index(arg)]
        fd(*data)

def save_universal_presets(filename, style, Value):
    with np.load(filename, 'r+', allow_pickle=True) as data:
        d = get_np(filename) #dict(data)
        td = {style: Value}
        d.update(td)
        np.savez(filename, **d)

def remove_universal_presets(filename, style, List):
    with np.load(filename, 'r+', allow_pickle=True) as data:
        d = get_np(filename) #dict(data)
        rs = [style, d[style]]
        List.append(rs)
        d.pop(style)
        np.savez(filename, **d)

def replace_removed_shader(filename, List):
    if not List:
        pass
    with np.load(filename, 'r+', allow_pickle=True) as data:
        d = get_np(filename) #dict(data)
        rl = List[-1]
        List.remove(rl)
        td = {rl[0]: rl[1]}
        d.update(td)
        np.savez(filename, **d)

def export_universal_presets(filePath, ext, style, Value): #'UN_Hair_'
    File = ext + style + '.npz'
    fp = get_filename(filePath, File)
    d = {style: Value}
    np.savez(fp, **d)


def import_universal_presets(filename, mport):
    fp = os.path.split(mport)[1]
    if fp.startswith('UN_Hshader_'):
        with np.load(filename, 'r+') as data:
            d = dict(data)
        with np.load(mport, 'r+') as imdata:
            imd = dict(imdata)
            d.update(imd)
        np.savez(filename, **d)
    else:
        pass

#######################################################################

def get_material(mat_name):
    mat = (bpy.data.materials.get(mat_name) or 
       bpy.data.materials.new(mat_name))
    return mat

def clear_material(object):
    object.data.materials.clear()

def clear_node(material):
    if material.node_tree:
        material.node_tree.links.clear()
        material.node_tree.nodes.clear()

def add_shader_node(material, node_type, Label, Name, location):
    nodes = material.node_tree.nodes
    new_node = nodes.new(type=node_type)
    new_node.label = Label
    new_node.name = Name
    new_node.location = location
    return new_node

def add_node_link(material, link1, link2):
    links = material.node_tree.links
    link = links.new(link1, link2)
    return link

def shader_prep(material):
    clear_material(bpy.context.object)
    clear_node(material)
    material.use_nodes = True
    clear_node(material)

#######################################################################
# Create new materials

def create_material(mat_name):
    material = get_material(mat_name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    if material.node_tree:
        material.node_tree.links.clear()
        material.node_tree.nodes.clear()

def material_setup(mat_name, info, links):
    material = get_material(mat_name)
    nodes = material.node_tree.nodes
    for i in info:
        add_shader_node(material, i[0], i[1], i[3], i[4])
    for i in links:
        add_node_link(material, eval(i[0]), eval(i[1]))

def new_material(mat_name, info, links):
    create_material(mat_name)
    material_setup(mat_name, info, links)

def set_material(mat_name, style):
        material = get_material(mat_name)
        nodes = material.node_tree.nodes
        style = np.array(style)
        args = style.tolist()
        for arg in args:
            id = nodes[arg[0]].bl_idname
            fd = func_dict(shader_set_dict(), id)
            data = args[args.index(arg)]
            fd(*data)

@timeit
def add_hair_shader():
    universal_hair_setup = [['ShaderNodeOutputMaterial', 'Material Output', 'Material Output', 'Material Output', (1150, 250)],
        ['ShaderNodeMixRGB', 'Mix', 'Gradient_Color', 'Gradient_Color', (-100, 280)], 
        ['ShaderNodeMixRGB', 'Mix', 'Tip_Color', 'Tip_Color', (150, 280)], 
        ['ShaderNodeMixRGB', 'Mix', 'Main_Color', 'Main_Color', (-325, 280)], 
        ['ShaderNodeHairInfo', 'Hair Info', 'Hair Info', 'Hair Info', (-890, 260)], 
        ['ShaderNodeAddShader', 'Add Shader', 'Highlight_Mix', 'Highlight_Mix', (680, 250)], 
        ['ShaderNodeBsdfDiffuse', 'Diffuse BSDF', 'Main_Diffuse', 'Main_Diffuse', (390, 240)], 
        ['ShaderNodeAddShader', 'Add Shader', 'Highlight_Mix_2', 'Highlight_Mix_2', (890, 260)], 
        ['ShaderNodeValToRGB', 'ColorRamp', 'Gradient_Control', 'Gradient_Control', (-660, 280)], 
        ['ShaderNodeValToRGB', 'ColorRamp', 'Main_Contrast', 'Main_Contrast', (-660, -5)], 
        ['ShaderNodeValToRGB', 'ColorRamp', 'Tip_Control', 'Tip_Control', (-660, 555)], 
        ['ShaderNodeBsdfGlossy', 'Glossy BSDF', 'Main_Highlight', 'Main_Highlight', (440, 80)], 
        ['ShaderNodeBsdfGlossy', 'Glossy BSDF', 'Secondary_Highlight', 'Secondary_Highlight', (680, 80)]]
    universal_hair_links = [['nodes["Gradient_Color"].outputs[0]', 'nodes["Tip_Color"].inputs[1]'], 
        ['nodes["Tip_Color"].outputs[0]', 'nodes["Main_Diffuse"].inputs[0]'], 
        ['nodes["Main_Color"].outputs[0]', 'nodes["Gradient_Color"].inputs[1]'], 
        ['nodes["Hair Info"].outputs[1]', 'nodes["Gradient_Control"].inputs[0]'], 
        ['nodes["Hair Info"].outputs[1]', 'nodes["Tip_Control"].inputs[0]'], 
        ['nodes["Hair Info"].outputs[4]', 'nodes["Main_Contrast"].inputs[0]'], 
        ['nodes["Highlight_Mix"].outputs[0]', 'nodes["Highlight_Mix_2"].inputs[0]'], 
        ['nodes["Main_Diffuse"].outputs[0]', 'nodes["Highlight_Mix"].inputs[0]'], 
        ['nodes["Highlight_Mix_2"].outputs[0]', 'nodes["Material Output"].inputs[0]'], 
        ['nodes["Gradient_Control"].outputs[0]', 'nodes["Gradient_Color"].inputs[0]'],
        ['nodes["Main_Contrast"].outputs[0]', 'nodes["Main_Color"].inputs[0]'], 
        ['nodes["Tip_Control"].outputs[0]', 'nodes["Tip_Color"].inputs[0]'], 
        ['nodes["Main_Highlight"].outputs[0]', 'nodes["Highlight_Mix"].inputs[1]'], 
        ['nodes["Secondary_Highlight"].outputs[0]', 'nodes["Highlight_Mix_2"].inputs[1]'],
        ['nodes["Highlight_Mix_2"].outputs[0]', 'nodes["Material Output"].inputs[0]']]
    universal_hair_default =  [
        ['Gradient_Color', 'MULTIPLY', True, 0.5, (0.466727614402771, 0.3782432973384857, 0.19663149118423462, 1.0), (0.2325773388147354, 0.15663157403469086, 0.07910887151956558, 1.0)],
        ['Tip_Color', 'SCREEN', True, 0.5, (0.466727614402771, 0.3782432973384857, 0.19663149118423462, 1.0), (0.38887712359428406, 0.28217148780822754, 0.10125808417797089, 1.0)],
        ['Main_Color', 'MIX', True, 0.5, (0.466727614402771, 0.3782432973384857, 0.19663149118423462, 1.0), (0.1809358447790146, 0.11345928907394409, 0.04037227854132652, 1.0)],
        ['Main_Diffuse', (0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0), 0.0, (0.0, 0.0, 0.0)], 
        ['Gradient_Control', [[0.045454543083906174, (1.0, 1.0, 1.0, 1.0)], [0.8909090161323547, (0.0, 0.0, 0.0, 1.0)]]], 
        ['Main_Contrast', [[0.08181827515363693, (0.0, 0.0, 0.0, 1.0)], [0.863636314868927, (1.0, 1.0, 1.0, 1.0)]]], 
        ['Tip_Control', [[0.5045454502105713, (0.0, 0.0, 0.0, 1.0)], [1.0, (1.0, 1.0, 1.0, 1.0)]]], 
        ['Main_Highlight', 'GGX', (0.08054633438587189, 0.0542692169547081, 0.030534733086824417, 1.0), 0.25, (0.0, 0.0, 0.0)], 
        ['Secondary_Highlight', 'GGX', (0.023630360141396523, 0.02180372178554535, 0.018096407875418663, 1.0), 0.15000000596046448, (0.0, 0.0, 0.0)]]
    material = get_material("Head_Hair")
    clear_node(material)
    material_setup("Head_Hair", universal_hair_setup, universal_hair_links)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    args = universal_hair_default
    for arg in args:
        id = nodes[arg[0]].bl_idname
        fd = func_dict(shader_set_dict(), id)
        data = args[args.index(arg)]
        fd(*data)

#######################################################################
# Node Ops

def node_info(nodes):
    return [[i.bl_idname, i.bl_label, i.label, i.name, i.location] for i in nodes[:]]


def set_links(material, nlink):
    for i in nlink:
        add_node_link(material, i[0], i[1])


#######################################################################

def shader_get_dict():
    shader_ = {
                'ShaderNodeMixRGB': get_mix_shader,
                'ShaderNodeValToRGB': get_colorramp_shader,
                'ShaderNodeBsdfDiffuse': get_bsdf_diffuse_shader,
                'ShaderNodeBsdfGlossy': get_bsdf_glossy_shader,
                'ShaderNodeBsdfHairPrincipled': get_hairP_shader,
                'ShaderNodeTexImage': get_image_shader,
            }
    return shader_

def shader_set_dict():
    shader_ = {
                'ShaderNodeMixRGB': set_mix_shader,
                'ShaderNodeValToRGB': set_colorramp_shader,
                'ShaderNodeBsdfDiffuse': set_bsdf_diffuse_shader,
                'ShaderNodeBsdfGlossy': set_bsdf_glossy_shader,
                'ShaderNodeBsdfHairPrincipled': set_hairP_shader,
                'ShaderNodeTexImage': set_image_shader,
            }
    return shader_

#######################################################################

def func_dict(Dict, Key):
    return Dict.get(Key, lambda: 'Invalid')


def get_all_shader_(nodes):
    info = node_info(nodes)
    setting = []
    for i in info:
        if i[0] in shader_get_dict():
            fd = func_dict(shader_get_dict(), i[0])
            setting.append(fd(nodes, i[3]))
    return setting

#######################################################################

# SHADER NODES

#######################################################################
# Hair Principled Shader

def get_hairP_shader(nodes, node_name):
    node = nodes.get(node_name)
    par = node.parametrization
    v5 = node.inputs[5].default_value
    v6 = node.inputs[6].default_value
    v7 = node.inputs[7].default_value
    v8 = node.inputs[8].default_value
    v9 = node.inputs[9].default_value
    v11 = node.inputs[11].default_value
    if par == 'COLOR':
        v0 = node.inputs[0].default_value[:]
        v1 = None
        v2 = None
        v3 = None
        v4 = None
        v10 = None
    elif par == 'MELANIN':
        v0 = None
        v1 = node.inputs[1].default_value
        v2 = node.inputs[2].default_value
        v3 = node.inputs[3].default_value[:]
        v4 = None
        v10 = node.inputs[10].default_value
    elif par == 'ABSORPTION':
        v0 = None
        v1 = None
        v2 = None
        v3 = None
        v4 = node.inputs[4].default_value[:]
        v10 = None
    return np.array([node_name, par, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11], dtype=object)


def set_hairP_shader(node_name, parametrization, v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11):
    material = bpy.context.object.active_material
    nodes = material.node_tree.nodes
    node = nodes.get(node_name)
    node.parametrization = parametrization #['ABSORPTION', 'COLOR', 'MELANIN']
    if parametrization is 'COLOR': #Direct Coloring
        node.inputs[0].default_value[:] = v0 #Color
    if parametrization is 'MELANIN': #Melanin Concetration
        node.inputs[1].default_value = v1 #Melanin
        node.inputs[2].default_value = v2 #Melanin Redness
        node.inputs[3].default_value = v3 #Tint
        node.inputs[10].default_value = v10 #Random Color
    if parametrization is 'ABSORPTION': #
        node.inputs[4].default_value[:] = v4 #Absorbtion Coefficient
    node.inputs[5].default_value = v5 #Roughness
    node.inputs[6].default_value = v6 #Radial Roughness
    node.inputs[7].default_value = v7 #Coat
    node.inputs[8].default_value = v8 #IOR
    node.inputs[9].default_value = v9 #offset
    node.inputs[11].default_value = v11 #Random Roughness

#######################################################################
# Color Ramp Shader

def get_colorramp_shader(nodes, node_name):
    node = nodes.get(node_name)
    dat = []
    for pos in node.color_ramp.elements[:]:
        p = pos.position
        col = pos.color[:]
        dat.append([p,col])
    return np.array([node_name, dat], dtype=object)

def set_colorramp_shader(node_name, p_c):
    material = bpy.context.object.active_material
    nodes = material.node_tree.nodes
    node = nodes.get(node_name)
    for i, pc in enumerate(p_c[3:]):
        node.color_ramp.elements[i].position = pc[0]
        node.color_ramp.elements[i].color[:] = pc[1]

#######################################################################
# Mix Shader

def get_mix_shader(nodes, node_name):
    node = nodes.get(node_name)
    blend = node.blend_type
    clamp = node.use_clamp
    fac = node.inputs[0].default_value
    col1 = node.inputs[1].default_value[:]
    col2 = node.inputs[2].default_value[:]
    return np.array([node_name, blend, clamp, fac, col1, col2], dtype=object)

def set_mix_shader(node_name, blend, clamp, fac, col1, col2):
    material = bpy.context.object.active_material
    nodes = material.node_tree.nodes
    node = nodes.get(node_name)
    node.blend_type = blend
    node.use_clamp = clamp
    node.inputs[0].default_value = fac
    node.inputs[1].default_value = col1
    node.inputs[2].default_value = col2

#######################################################################
# BSDF Diffuse

def get_bsdf_diffuse_shader(nodes, node_name):
    node = nodes.get(node_name)
    col = node.inputs[0].default_value[:]
    rough = node.inputs[1].default_value
    norm = node.inputs[2].default_value[:]
    return np.array([node_name, col, rough, norm], dtype=object)

def set_bsdf_diffuse_shader(node_name, col, rough, norm):
    material = bpy.context.object.active_material
    nodes = material.node_tree.nodes
    node = nodes.get(node_name)
    node.inputs[0].default_value = col
    node.inputs[1].default_value = rough
    node.inputs[2].default_value = norm

#######################################################################
# BSDF Glossy

def get_bsdf_glossy_shader(nodes, node_name):
    node = nodes.get(node_name)
    distribution = node.distribution
    col = node.inputs[0].default_value[:]
    rough = node.inputs[1].default_value
    norm = node.inputs[2].default_value[:]
    return np.array([node_name, distribution, col, rough, norm], dtype=object)

def set_bsdf_glossy_shader(node_name, distribution, col, rough, norm):
    material = bpy.context.object.active_material
    nodes = material.node_tree.nodes
    node = nodes.get(node_name)
    node.distribution = distribution
    node.inputs[0].default_value = col
    node.inputs[1].default_value = rough
    node.inputs[2].default_value = norm

#######################################################################
# Add ShaderImage Texture

def get_image_shader(nodes, node_name):
    node = nodes.get(node_name)
    texture = node.image
    col = node.color
    ext = node.extension
    bc = node.color_mapping.blend_color
    bf = node.color_mapping.blend_factor
    bt = node.color_mapping.blend_type
    bright = node.color_mapping.brightness
    contrast = node.color_mapping.contrast
    saturation = node.color_mapping.saturation
    ucr = node.color_mapping.use_color_ramp
    cm1 = node.color_mapping.color_ramp.color_mode
    alpha1 = node.color_mapping.color_ramp.elements[0].alpha
    col1 = node.color_mapping.color_ramp.elements[0].color[:]
    cpos1 = node.color_mapping.color_ramp.elements[0].position
    cm2 = node.color_mapping.color_ramp.color_mode
    alpha2 = node.color_mapping.color_ramp.elements[1].alpha
    col2 = node.color_mapping.color_ramp.elements[1].color[:]
    cpos2 = node.color_mapping.color_ramp.elements[1].position
    h_int = node.color_mapping.color_ramp.hue_interpolation
    intp = node.color_mapping.color_ramp.interpolation
    fs = node.image_user.frame_start
    fc = node.image_user.frame_current
    fd = node.image_user.frame_duration
    fo = node.image_user.frame_offset
    uaf = node.image_user.use_auto_refresh
    ucy = node.image_user.use_cyclic
    return np.array([node_name, texture, col, ext, bc, bf, bt, bright, contrast, saturation, ucr, cm1, alpha1, col1, cpos1, cm2, alpha2, col2, cpos2, h_int, intp, fs, fc, fd, fo, uaf, ucy], dtype=object)

def set_image_shader(node_name, texture, col, ext, bc, bf, bt, bright, contrast, saturation, ucr, cm1, alpha1, col1, cpos1, cm2, alpha2, col2, cpos2, h_int, intp, fs, fc, fd, fo, uaf, ucy):
    material = bpy.context.object.active_material
    nodes = material.node_tree.nodes
    node = nodes.get(node_name)
    node.image = texture
    node.extension = ext
    node.color_mapping.blend_color = bc
    node.color_mapping.blend_factor = bf
    node.color_mapping.blend_type = bt
    node.color_mapping.brightness = bright
    node.color_mapping.contrast = contrast
    node.color_mapping.saturation = saturation
    node.color_mapping.use_color_ramp = ucr
    node.color_mapping.color_ramp.color_mode = cm1
    node.color_mapping.color_ramp.elements[0].alpha = alpha1
    node.color_mapping.color_ramp.elements[0].color[:] = col1
    node.color_mapping.color_ramp.elements[0].position = cpos1
    node.color_mapping.color_ramp.color_mode = cm2
    node.color_mapping.color_ramp.elements[1].alpha = alpha2
    node.color_mapping.color_ramp.elements[1].color[:] = col2
    node.color_mapping.color_ramp.elements[1].position = cpos2
    node.color_mapping.color_ramp.hue_interpolation = h_int
    node.color_mapping.color_ramp.interpolation = intp
    node.image_user.frame_start = fs
    node.image_user.frame_current = fc
    node.image_user.frame_duration = fd
    node.image_user.frame_offset = fo
    node.image_user.use_auto_refresh = uaf
    node.image_user.use_cyclic = ucy


#######################################################################
@timeit
class HairEngine:
    '''
    '''
    def __init__(self, ob):
        self.object = ob
        self.armature = self.object.parent
        self.hair = name_verify("Hair_Particle", "Hair_Accessories")
        self.collection = "Hair_Accessories"
        self.view_hide = self.object.hide_viewport
        self.render_hide = self.object.hide_render
        self.material = get_material(self.hair)
        self.universal_hair_setup = [['ShaderNodeOutputMaterial', 'Material Output', 'Material Output', 'Material Output', (1150, 250)],
            ['ShaderNodeMixRGB', 'Mix', 'Gradient_Color', 'Gradient_Color', (-100, 280)], 
            ['ShaderNodeMixRGB', 'Mix', 'Tip_Color', 'Tip_Color', (150, 280)], 
            ['ShaderNodeMixRGB', 'Mix', 'Main_Color', 'Main_Color', (-325, 280)], 
            ['ShaderNodeHairInfo', 'Hair Info', 'Hair Info', 'Hair Info', (-890, 260)], 
            ['ShaderNodeAddShader', 'Add Shader', 'Highlight_Mix', 'Highlight_Mix', (680, 250)], 
            ['ShaderNodeBsdfDiffuse', 'Diffuse BSDF', 'Main_Diffuse', 'Main_Diffuse', (390, 240)], 
            ['ShaderNodeAddShader', 'Add Shader', 'Highlight_Mix_2', 'Highlight_Mix_2', (890, 260)], 
            ['ShaderNodeValToRGB', 'ColorRamp', 'Gradient_Control', 'Gradient_Control', (-660, 280)], 
            ['ShaderNodeValToRGB', 'ColorRamp', 'Main_Contrast', 'Main_Contrast', (-660, -5)], 
            ['ShaderNodeValToRGB', 'ColorRamp', 'Tip_Control', 'Tip_Control', (-660, 555)], 
            ['ShaderNodeBsdfGlossy', 'Glossy BSDF', 'Main_Highlight', 'Main_Highlight', (440, 80)], 
            ['ShaderNodeBsdfGlossy', 'Glossy BSDF', 'Secondary_Highlight', 'Secondary_Highlight', (680, 80)]]
        self.universal_hair_links = [['nodes["Gradient_Color"].outputs[0]', 'nodes["Tip_Color"].inputs[1]'], 
            ['nodes["Tip_Color"].outputs[0]', 'nodes["Main_Diffuse"].inputs[0]'], 
            ['nodes["Main_Color"].outputs[0]', 'nodes["Gradient_Color"].inputs[1]'], 
            ['nodes["Hair Info"].outputs[1]', 'nodes["Gradient_Control"].inputs[0]'], 
            ['nodes["Hair Info"].outputs[1]', 'nodes["Tip_Control"].inputs[0]'], 
            ['nodes["Hair Info"].outputs[4]', 'nodes["Main_Contrast"].inputs[0]'], 
            ['nodes["Highlight_Mix"].outputs[0]', 'nodes["Highlight_Mix_2"].inputs[0]'], 
            ['nodes["Main_Diffuse"].outputs[0]', 'nodes["Highlight_Mix"].inputs[0]'], 
            ['nodes["Highlight_Mix_2"].outputs[0]', 'nodes["Material Output"].inputs[0]'], 
            ['nodes["Gradient_Control"].outputs[0]', 'nodes["Gradient_Color"].inputs[0]'],
            ['nodes["Main_Contrast"].outputs[0]', 'nodes["Main_Color"].inputs[0]'], 
            ['nodes["Tip_Control"].outputs[0]', 'nodes["Tip_Color"].inputs[0]'], 
            ['nodes["Main_Highlight"].outputs[0]', 'nodes["Highlight_Mix"].inputs[1]'], 
            ['nodes["Secondary_Highlight"].outputs[0]', 'nodes["Highlight_Mix_2"].inputs[1]'],
            ['nodes["Highlight_Mix_2"].outputs[0]', 'nodes["Material Output"].inputs[0]']]
        self.universal_hair_default =  [
            ['Gradient_Color', 'MULTIPLY', True, 0.5, (0.466727614402771, 0.3782432973384857, 0.19663149118423462, 1.0), (0.2325773388147354, 0.15663157403469086, 0.07910887151956558, 1.0)],
            ['Tip_Color', 'SCREEN', True, 0.5, (0.466727614402771, 0.3782432973384857, 0.19663149118423462, 1.0), (0.38887712359428406, 0.28217148780822754, 0.10125808417797089, 1.0)],
            ['Main_Color', 'MIX', True, 0.5, (0.466727614402771, 0.3782432973384857, 0.19663149118423462, 1.0), (0.1809358447790146, 0.11345928907394409, 0.04037227854132652, 1.0)],
            ['Main_Diffuse', (0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0), 0.0, (0.0, 0.0, 0.0)], 
            ['Gradient_Control', [[0.045454543083906174, (1.0, 1.0, 1.0, 1.0)], [0.8909090161323547, (0.0, 0.0, 0.0, 1.0)]]], 
            ['Main_Contrast', [[0.08181827515363693, (0.0, 0.0, 0.0, 1.0)], [0.863636314868927, (1.0, 1.0, 1.0, 1.0)]]], 
            ['Tip_Control', [[0.5045454502105713, (0.0, 0.0, 0.0, 1.0)], [1.0, (1.0, 1.0, 1.0, 1.0)]]], 
            ['Main_Highlight', 'GGX', (0.08054633438587189, 0.0542692169547081, 0.030534733086824417, 1.0), 0.25, (0.0, 0.0, 0.0)], 
            ['Secondary_Highlight', 'GGX', (0.023630360141396523, 0.02180372178554535, 0.018096407875418663, 1.0), 0.15000000596046448, (0.0, 0.0, 0.0)]]
        self.principled_hair_setup = [['ShaderNodeOutputMaterial', 'Material Output', 'Material Output', 'Material Output', (400,100)],
            ['ShaderNodeBsdfHairPrincipled', 'Principled Hair BSDF', 'Hair_Shader', 'Hair_Shader', (100,100)]]
        self.principled_hair_links = [['nodes["Hair_Shader"].outputs[0]', 'nodes["Material Output"].inputs[0]']]
        self.principled_hair_default = [['Hair_Shader', 'MELANIN', None, 0.11400000005960464, 0.3, (1.0, 0.52252197265625, 0.52252197265625, 1.0), None, 0.5, 0.0, 0.0, 1.4500000476837158, 0.03490658476948738, 0.0, 0.0]]
    '''
    '''
    #Add particle hair
    def add_hair(self):
        p_sys = bpy.context.object.modifiers.new("hair", 'PARTICLE_SYSTEM').particle_system
        p_sys.settings.type = 'HAIR'
        p_sys.settings.child_type = 'INTERPOLATED'
        p_sys.settings.hair_length = 0.2
        p_sys.settings.root_radius = 0.03
        p_sys.settings.count = 1000
        p_sys.settings.hair_step = 5
        p_sys.settings.child_nbr = 20
        p_sys.settings.rendered_child_count = 20
        p_sys.settings.child_length = 0.895
        bpy.context.object.show_instancer_for_viewport = False
        bpy.context.object.show_instancer_for_render = False
        bpy.ops.particle.connect_hair(all=True)
    '''
    '''
    def hair_armature_mod(self, hair, vertgroup):
        new = "ARM_{}".format(hair)
        a_mod = bpy.context.object.modifiers.new(new, 'ARMATURE')
        a_mod.object = self.armature
        a_mod.vertex_group = vertgroup #'head'
    '''
    '''
    def make_hair(self, target):
        obj = bpy.data.objects
        coll = bpy.context.collection
        get_collection(self.collection)
        hair = CopyObject(bpy.context.object, self.hair).new_object_full()
        bpy.data.collections[self.collection].objects.link(obj[self.hair])
        coll.objects.unlink(obj[self.hair])
        self.add_hair()
        material = self.material
        material.use_nodes = True
        nodes = material.node_tree.nodes
        clear_node(material)
        for i in self.universal_hair_setup:
            add_shader_node(material, i[0], i[2], i[3], i[4])
        for i in self.universal_hair_links:
            add_node_link(material, eval(i[0]), eval(i[1]))
        obj[self.hair].data.materials.append(material)
        args = self.universal_hair_default
        for arg in args:
            id = nodes[arg[0]].bl_idname
            fd = func_dict(shader_set_dict(), id)
            data = args[args.index(arg)]
            fd(*data)
        self.hair_armature_mod(self.hair, target)

@timeit
def hair_from_selected():
    heng = HairEngine(bpy.context.object)
    heng.make_hair('head')

@timeit
def convert_to_curve(ob):
    cl = bpy.data.collections
    col =  bpy.context.collection
    coll = get_obj_collection(ob)
    h_acc = get_collection("Hair_Accessories")
    obj = bpy.data.objects
    hc_name = name_verify("Hair_Cards", "Hair_Cards")
    h_c = get_subcollection(hc_name, "Hair_Accessories")
    add_emp = lambda l: bpy.ops.object.empty_add(type="SPHERE", radius=0.1, align='WORLD', location=l, rotation=(0,0,0))
    def_image = bpy.data.images.load(os.path.join(get_hair_dir(), "Sample_01.png"), check_existing=True)
    def_disp = bpy.data.images.load(os.path.join(get_hair_dir(), "DisplacementMap.png"), check_existing=True)
    hair_card_setup = [['ShaderNodeMixRGB', 'Mix', 'Gradient_Color', 'Gradient_Color', (-100.0, 280.0)], 
            ['ShaderNodeMixRGB', 'Mix', 'Tip_Color', 'Tip_Color', (150.0, 280.0)], 
            ['ShaderNodeMixRGB', 'Mix', 'Main_Color', 'Main_Color', (-325.0, 280.0)], 
            ['ShaderNodeHairInfo', 'Hair Info', 'Hair Info', 'Hair Info', (-890.0, 260.0)], 
            ['ShaderNodeAddShader', 'Add Shader', 'Highlight_Mix', 'Highlight_Mix', (680.0, 250.0)], 
            ['ShaderNodeBsdfDiffuse', 'Diffuse BSDF', 'Main_Diffuse', 'Main_Diffuse', (390.0, 240.0)], 
            ['ShaderNodeValToRGB', 'ColorRamp', 'Gradient_Control', 'Gradient_Control', (-660.0, 280.0)], 
            ['ShaderNodeValToRGB', 'ColorRamp', 'Main_Contrast', 'Main_Contrast', (-660.0, -5.0)], 
            ['ShaderNodeValToRGB', 'ColorRamp', 'Tip_Control', 'Tip_Control', (-660.0, 555.0)], 
            ['ShaderNodeBsdfGlossy', 'Glossy BSDF', 'Main_Highlight', 'Main_Highlight', (440.0, 80.0)], 
            ['ShaderNodeBsdfGlossy', 'Glossy BSDF', 'Secondary_Highlight', 'Secondary_Highlight', (680.0, 80.0)], 
            ['ShaderNodeAddShader', 'Add Shader', 'Highlight_Mix_2', 'Highlight_Mix_2', (890.0, 260.0)], 
            ['ShaderNodeTexImage', 'Image Texture', 'Hair_Alpha', 'Hair_Alpha', (165.0, 635.0)], 
            ['ShaderNodeBsdfDiffuse', 'Diffuse BSDF', 'Hair_Diffuse', 'Hair_Diffuse', (515.0, 400.0)], 
            ['ShaderNodeBsdfTransparent', 'Transparent BSDF', 'Hair_Transparency', 'Hair_Transparency', (510.0, 500.0338439941406)], 
            ['ShaderNodeMixShader', 'Mix Shader', 'Diffuse_Mix', 'Diffuse_Mix', (750.0, 550.0)], 
            ['ShaderNodeMixShader', 'Mix Shader', 'Color_Mix', 'Color_Mix', (1030.0, 615.0)], 
            ['ShaderNodeOutputMaterial', 'Material Output', 'Material Output', 'Material Output', (1175.0, 280.0)],
            ['ShaderNodeTexImage', 'Image Texture', 'Hair_Displacement', 'Hair_Displacement', (902.8016967773438, 90.0456771850586)]]
    hair_card_links = [['nodes["Gradient_Color"].outputs[0]', 'nodes["Tip_Color"].inputs[1]'], 
            ['nodes["Tip_Color"].outputs[0]', 'nodes["Main_Diffuse"].inputs[0]'], 
            ['nodes["Main_Color"].outputs[0]', 'nodes["Gradient_Color"].inputs[1]'], 
            ['nodes["Hair Info"].outputs[1]', 'nodes["Gradient_Control"].inputs[0]'], 
            ['nodes["Hair Info"].outputs[1]', 'nodes["Tip_Control"].inputs[0]'], 
            ['nodes["Hair Info"].outputs[4]', 'nodes["Main_Contrast"].inputs[0]'], 
            ['nodes["Highlight_Mix"].outputs[0]', 'nodes["Highlight_Mix_2"].inputs[0]'], 
            ['nodes["Main_Diffuse"].outputs[0]', 'nodes["Highlight_Mix"].inputs[0]'], 
            ['nodes["Gradient_Control"].outputs[0]', 'nodes["Gradient_Color"].inputs[0]'], 
            ['nodes["Main_Contrast"].outputs[0]', 'nodes["Main_Color"].inputs[0]'], 
            ['nodes["Tip_Control"].outputs[0]', 'nodes["Tip_Color"].inputs[0]'], 
            ['nodes["Main_Highlight"].outputs[0]', 'nodes["Highlight_Mix"].inputs[1]'], 
            ['nodes["Secondary_Highlight"].outputs[0]', 'nodes["Highlight_Mix_2"].inputs[1]'], 
            ['nodes["Highlight_Mix_2"].outputs[0]', 'nodes["Color_Mix"].inputs[2]'], 
            ['nodes["Hair_Alpha"].outputs[0]', 'nodes["Hair_Diffuse"].inputs[0]'], 
            ['nodes["Hair_Alpha"].outputs[1]', 'nodes["Diffuse_Mix"].inputs[0]'], 
            ['nodes["Hair_Alpha"].outputs[1]', 'nodes["Color_Mix"].inputs[0]'], 
            ['nodes["Hair_Diffuse"].outputs[0]', 'nodes["Diffuse_Mix"].inputs[2]'], 
            ['nodes["Hair_Transparency"].outputs[0]', 'nodes["Diffuse_Mix"].inputs[1]'], 
            ['nodes["Diffuse_Mix"].outputs[0]', 'nodes["Color_Mix"].inputs[1]'], 
            ['nodes["Color_Mix"].outputs[0]', 'nodes["Material Output"].inputs[0]'],
            ['nodes["Hair_Displacement"].outputs[0]', 'nodes["Material Output"].inputs[2]']]
    hair_card_default = [['Hair_Alpha', def_image, (0.6079999804496765, 0.6079999804496765, 0.6079999804496765), 'REPEAT', (0.800000011920929, 0.800000011920929, 0.800000011920929), 0.0, 'MIX', 1.0, 1.0, 1.0, False, 'RGB', 1.0, (0.0, 0.0, 0.0, 1.0), 0.0, 'RGB', 1.0, (1.0, 1.0, 1.0, 1.0), 1.0, 'NEAR', 'LINEAR', 1, 1, 1, 0, False, False],
            ['Hair_Displacement', def_disp, (0.6079999804496765, 0.6079999804496765, 0.6079999804496765), 'REPEAT', (0.800000011920929, 0.800000011920929, 0.800000011920929), 0.0, 'MIX', 1.0, 1.0, 1.0, False, 'RGB', 1.0, (0.0, 0.0, 0.0, 1.0), 0.0, 'RGB', 1.0, (1.0, 1.0, 1.0, 1.0), 1.0, 'NEAR', 'LINEAR', 1, 0, 1, -1, False, False],
            ['Gradient_Color', 'MULTIPLY', True, 0.5, (0.466727614402771, 0.3782432973384857, 0.19663149118423462, 1.0), (0.2325773388147354, 0.15663157403469086, 0.07910887151956558, 1.0)],
            ['Tip_Color', 'SCREEN', True, 0.5, (0.466727614402771, 0.3782432973384857, 0.19663149118423462, 1.0), (0.38887712359428406, 0.28217148780822754, 0.10125808417797089, 1.0)],
            ['Main_Color', 'MIX', True, 0.5, (0.466727614402771, 0.3782432973384857, 0.19663149118423462, 1.0), (0.1809358447790146, 0.11345928907394409, 0.04037227854132652, 1.0)],
            ['Main_Diffuse', (0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0), 0.0, (0.0, 0.0, 0.0)], 
            ['Gradient_Control', [[0.045454543083906174, (1.0, 1.0, 1.0, 1.0)], [0.8909090161323547, (0.0, 0.0, 0.0, 1.0)]]], 
            ['Main_Contrast', [[0.08181827515363693, (0.0, 0.0, 0.0, 1.0)], [0.863636314868927, (1.0, 1.0, 1.0, 1.0)]]], 
            ['Tip_Control', [[0.5045454502105713, (0.0, 0.0, 0.0, 1.0)], [1.0, (1.0, 1.0, 1.0, 1.0)]]], 
            ['Main_Highlight', 'GGX', (0.08054633438587189, 0.0542692169547081, 0.030534733086824417, 1.0), 0.25, (0.0, 0.0, 0.0)], 
            ['Secondary_Highlight', 'GGX', (0.023630360141396523, 0.02180372178554535, 0.018096407875418663, 1.0), 0.15000000596046448, (0.0, 0.0, 0.0)]]
    active_ob(ob, None)
    bpy.ops.object.modifier_convert(modifier='hair')
    bpy.ops.object.convert(target='CURVE')
    bpy.context.object.data.extrude = 0.01
    bpy.ops.object.editmode_toggle()
    bpy.ops.curve.select_all(action='SELECT')
    bpy.ops.transform.tilt(value=-1.5708, mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
    bpy.context.object.data.use_uv_as_generated = True
    bpy.ops.curve.match_texture_space()
    bpy.ops.curve.handle_type_set(type='ALIGNED')
    bpy.ops.object.editmode_toggle()
    bpy.context.object.name = hc_name
    bpy.context.object.data.name = hc_name
    material = get_material(hc_name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    clear_node(material)
    for i in hair_card_setup:
        add_shader_node(material, i[0], i[2], i[3], i[4])
    for i in hair_card_links:
        add_node_link(material, eval(i[0]), eval(i[1]))
    obj[hc_name].data.materials.append(material)
    args = hair_card_default
    for arg in args:
        id = nodes[arg[0]].bl_idname
        fd = func_dict(shader_set_dict(), id)
        data = args[args.index(arg)]
        fd(*data)
    material.blend_method = 'CLIP'
    material.shadow_method = 'OPAQUE'
    material.pass_index = 32
    try:
        col.objects.unlink(obj[hc_name])
    except:
        pass
    try:
        cl[coll].objects.unlink(obj[hc_name])
    except:
        pass
    h_c.objects.link(obj[hc_name])
    if "Hair_Parent" not in [i.name for i in obj]:
        rm90 = rotation_matrix(np.radians(-90), 0, 0)
        rm90_4x4 = Matrix(rm90).to_4x4()
        arm = [i for i in bpy.data.objects if i.type == 'ARMATURE'][0]
        bmt = arm.pose.bones['head'].matrix.copy()
        blc = arm.pose.bones['head'].head
        add_emp(blc)
        bpy.context.object.name = "Hair_Parent"
        h_p = obj["Hair_Parent"]
        h_p.matrix_world = (bmt @ rm90_4x4)
        active_ob("Hair_Parent", None)
        bpy.ops.object.constraint_add(type='COPY_TRANSFORMS')
        bpy.context.object.constraints["Copy Transforms"].target = arm
        bpy.context.object.constraints["Copy Transforms"].subtarget = "head"
        h_acc.objects.link(h_p)
        col.objects.unlink(h_p)
    bpy.ops.object.select_all(action='DESELECT')
    active_ob("Hair_Parent", [hc_name])
    bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
    bpy.ops.object.select_all(action='DESELECT')
    hide_ob(ob, True)
    hide_ob("Hair_Parent", True)
    bpy.ops.object.select_all(action='DESELECT')
    active_ob(hc_name, None)

@timeit
def seperate_cards(ob):
    sep = "{}_Seperated".format(ob)
    h_acc = get_collection("Hair_Accessories")
    h_c = get_subcollection("Hair_Cards", "Hair_Accessories")
    h_cs = get_subcollection(sep, "Hair_Accessories")
    obj = bpy.data.objects
    co = get_all_curve_points(ob)
    curves = []
    for i, v in enumerate(co):
        co = v
        count = len(co)
        n = "{}_CC{}".format(ob, i)
        curves.append(n)
        nc = bpy.data.curves.new(n, type='CURVE')
        cob = bpy.data.objects.new(n, nc)
        curve = obj[n]
        h_cs.objects.link(curve)
        nc.dimensions = '3D'
        bez = nc.splines.new('BEZIER')
        bz = bez.bezier_points
        bz.add((count - 1))
        bz.foreach_set("co", co.ravel())
        for i, p in enumerate(co):
            bz[i].handle_left_type="AUTO"
            bz[i].handle_right_type="AUTO"
        curve.data.extrude = obj[ob].data.extrude
        bpy.ops.object.editmode_toggle()
        bpy.ops.curve.select_all(action='SELECT')
        bpy.context.object.data.use_uv_as_generated = True
        bpy.context.object.data.use_auto_texspace = True
        bpy.ops.curve.handle_type_set(type='ALIGNED')
        # bpy.ops.transform.tilt(value=1.5708, mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
        bpy.ops.object.editmode_toggle()
        curve.data.materials.append(obj[ob].active_material)
    hide_ob(ob, True)
    bpy.ops.object.select_all(action='DESELECT')
    hide_ob("Hair_Parent", False)
    active_ob("Hair_Parent", curves)
    bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
    bpy.ops.object.select_all(action='DESELECT')
    hide_ob("Hair_Parent", True)

@timeit
def convert_to_mesh(ob):
    obj = bpy.data.objects
    hc_name = "{}_Mesh".format(ob)
    active_ob(ob, None)
    bpy.ops.object.convert(target='MESH', keep_original=True)
    bpy.context.object.name = hc_name
    bpy.context.object.data.name = hc_name
    active_ob(hc_name, None)
    hide_ob("Hair_Parent", False)
    active_ob("Hair_Parent", [hc_name])
    bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
    bpy.ops.object.select_all(action='DESELECT')
    hide_ob(ob, True)
    hide_ob("Hair_Parent", True)
    active_ob(hc_name, None)

@timeit
def curve_armature(ob):
    ccoll = bpy.context.collection
    h_acc = get_collection("Hair_Accessories")
    tar = "{}_Targets".format(ob)
    h_tar = get_subcollection(tar, "Hair_Accessories")
    add_emp = lambda l, r: bpy.ops.object.empty_add(type='CUBE', radius=0.005, align='WORLD', location=l, rotation=r)
    e_name = "{}_Edge".format(ob)
    m_name = "{}_Mesh".format(ob)
    a_name = "{}_Armature".format(ob)
    obj = bpy.data.objects
    curve = obj[ob]
    width = curve.data.extrude
    active_ob(ob, None)
    bpy.ops.object.convert(target='MESH', keep_original=True)
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.follow_active_quads(mode='LENGTH_AVERAGE')
    bpy.ops.object.editmode_toggle()
    bpy.context.object.name = m_name
    bpy.context.object.data.name = m_name
    obj[m_name].data.materials.append(obj[ob].active_material)
    active_ob(m_name, None)
    set_hc_uv(m_name)
    active_ob(ob, None)
    curve.data.extrude = 0
    bpy.ops.object.convert(target='MESH', keep_original=True)
    bpy.context.object.name = e_name
    bpy.context.object.data.name = e_name
    curve.data.extrude = width
    ed = obj[e_name]
    vt = ed.data.vertices
    count = len(vt)
    eco = np.empty(count * 3)
    vt.foreach_get("co", eco)
    eco.shape = (count, 3)
    vg = ed.vertex_groups
    nvg = vg.new(name="pin")
    index = np.arange(count).ravel().tolist()
    nvg.add(index, 1.0, 'REPLACE')
    wts = np.linspace(1.0, .1, num=count).tolist()
    for i, w in enumerate(wts):
        vt[i].groups[0].weight = w
    emod = ed.modifiers.new("Cloth", 'CLOTH')
    ed.modifiers['Cloth'].settings.vertex_group_mass = "pin"
    add_armature(a_name, h_acc.name)
    active_ob(a_name, None)
    eidx = edge_idx_chain(count)
    bones = []
    for i, v in enumerate(eidx):
        n = "{}_B{}".format(ob, i)
        add_bone(a_name, n, eco[v])
        bones.append(n)
    bones.reverse()
    arm = obj[a_name]
    pb = arm.pose.bones
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
    eb = arm.data.edit_bones
    for idx, b in enumerate(bones[:-1]):
        ix = idx + 1
        eb[b].parent = eb[bones[ix]]
        eb[b].use_connect = True
    bpy.ops.object.mode_set(mode='OBJECT')
    targets = []
    bones.reverse()
    for i, v in enumerate(eco[:-1]):
        r = pb[bones[i]].matrix.to_euler('XYZ')
        nm = "{}_Target_{}".format(ob, i)
        add_emp(v, r)
        obj["Empty"].name = nm
        h_tar.objects.link(obj[nm])
        ccoll.objects.unlink(obj[nm])
        targets.append(nm)
    tidx = count - 1
    nml = "{}_Target_{}".format(ob, tidx)
    rl = pb[bones[-1]].matrix.to_euler('XYZ')
    add_emp(eco[tidx], rl)
    obj["Empty"].name = nml
    h_tar.objects.link(obj[nml])
    ccoll.objects.unlink(obj[nml])
    targets.append(nml)
    for ix, k in enumerate(targets):
        active_ob(e_name, [k])
        bpy.ops.object.mode_set(mode='EDIT', toggle=False)
        par = bpy.data.objects[e_name]
        ch = bpy.data.objects[k]
        ch.parent = par
        ch.matrix_world = par.matrix_world
        ch.parent_type = 'VERTEX'
        ch.parent_vertices[0] = ix
        bpy.ops.object.mode_set(mode='OBJECT')
    active_ob(a_name, None)
    for b, t in list(zip(bones, targets[1:])):
        bct = pb[b].constraints.new('DAMPED_TRACK')
        bct.target = obj[t]
    bpy.ops.object.select_all(action='DESELECT')
    curve.data.extrude = width
    active_ob(a_name, [m_name])
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    bpy.ops.object.select_all(action='DESELECT')
    hide_ob("Hair_Parent", False)
    active_ob("Hair_Parent", [e_name, a_name])
    bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
    bpy.ops.object.select_all(action='DESELECT')
    hide_ob(ob, True)
    hide_ob(e_name, True)
    hide_ob(a_name, True)
    hide_ob("Hair_Parent", True)
    for t in targets:
        hide_ob(t, True)
    active_ob(m_name, None)

def edit_weights():
    obj = bpy.data.objects
    mn = bpy.context.object.name
    o = mn[:-5]
    e = "{}_Edge".format(o)
    obj[e].hide_viewport = False
    active_ob(e, None)
    bpy.ops.object.editmode_toggle()

def finish_weights():
    obj = bpy.data.objects
    en = bpy.context.object.name
    o = en[:-5]
    m = "{}_Mesh".format(o)
    obj[en].hide_viewport = True
    active_ob(m, None)

#---------------------------------------------------------

@timeit
def add_gp():
    obj = bpy.data.objects
    cl = bpy.data.collections
    col =  bpy.context.collection
    h_acc = get_collection("Hair_Accessories")
    h_c = get_subcollection("Hair_GP", "Hair_Accessories")
    n = name_verify("Hair_Card_GP", "Hair_GP")
    # arm = [i for i in bpy.data.objects if i.type == 'ARMATURE'][0]
    # blc = arm.pose.bones['head'].head
    bpy.ops.object.gpencil_add(location=(0,0,0), type='EMPTY')
    bpy.context.object.name = n
    h_c.objects.link(obj[n])
    col.objects.unlink(obj[n])
    bpy.ops.gpencil.paintmode_toggle()

def get_gp_co(object, layer, frame, stroke):
    gp = object.data.layers[layer].frames[frame].strokes[stroke]
    points = gp.points
    co = np.empty((len(points)*3))
    points.foreach_get('co', co)
    co.shape = ((len(points), 3))
    return co

@timeit
def get_all_gp_co(object, layer, frame):
    gp = object.data.layers[layer].frames[frame].strokes
    count = len(gp)
    allcos = []
    for stroke in gp:
        points = stroke.points
        co = np.empty((len(points)*3))
        points.foreach_get('co', co)
        co.shape = ((len(points), 3))
        allcos.append(co)
    return allcos

@timeit
def gp_simplify(ob, dist):
    active_ob(ob, None)
    bpy.ops.object.gpencil_modifier_add(type='GP_SIMPLIFY')
    bpy.context.object.grease_pencil_modifiers["Simplify"].mode = 'MERGE'
    bpy.context.object.grease_pencil_modifiers["Simplify"].distance = dist
    bpy.ops.object.gpencil_modifier_apply(apply_as='DATA', modifier="Simplify")

@timeit
class GreaseHairCards:
    '''
    '''
    def __init__(self, ob, width):
        self.grease_pencil = ob
        self.object = bpy.data.objects[self.grease_pencil]
        self.co = get_all_gp_co(self.object, 0, 0)
        self.hair_card = name_check("Hair_Card")
        self.hair_object = "{}_Mesh".format(self.hair_card)
        self.cloth_control = "{}_Edge".format(self.hair_card)
        self.v_group = "pin"
        self.collection = "Hair_Accessories"
        self.width = width
        self.hair_armature = "{}_Armature".format(self.hair_card)
        self.def_image = bpy.data.images.load(os.path.join(get_hair_dir(), "Sample_01.png"), check_existing=True)
        self.def_disp = bpy.data.images.load(os.path.join(get_hair_dir(), "DisplacementMap.png"), check_existing=True)
        self.hair_card_setup = [['ShaderNodeMixRGB', 'Mix', 'Gradient_Color', 'Gradient_Color', (-100.0, 280.0)], 
            ['ShaderNodeMixRGB', 'Mix', 'Tip_Color', 'Tip_Color', (150.0, 280.0)], 
            ['ShaderNodeMixRGB', 'Mix', 'Main_Color', 'Main_Color', (-325.0, 280.0)], 
            ['ShaderNodeHairInfo', 'Hair Info', 'Hair Info', 'Hair Info', (-890.0, 260.0)], 
            ['ShaderNodeAddShader', 'Add Shader', 'Highlight_Mix', 'Highlight_Mix', (680.0, 250.0)], 
            ['ShaderNodeBsdfDiffuse', 'Diffuse BSDF', 'Main_Diffuse', 'Main_Diffuse', (390.0, 240.0)], 
            ['ShaderNodeValToRGB', 'ColorRamp', 'Gradient_Control', 'Gradient_Control', (-660.0, 280.0)], 
            ['ShaderNodeValToRGB', 'ColorRamp', 'Main_Contrast', 'Main_Contrast', (-660.0, -5.0)], 
            ['ShaderNodeValToRGB', 'ColorRamp', 'Tip_Control', 'Tip_Control', (-660.0, 555.0)], 
            ['ShaderNodeBsdfGlossy', 'Glossy BSDF', 'Main_Highlight', 'Main_Highlight', (440.0, 80.0)], 
            ['ShaderNodeBsdfGlossy', 'Glossy BSDF', 'Secondary_Highlight', 'Secondary_Highlight', (680.0, 80.0)], 
            ['ShaderNodeAddShader', 'Add Shader', 'Highlight_Mix_2', 'Highlight_Mix_2', (890.0, 260.0)], 
            ['ShaderNodeTexImage', 'Image Texture', 'Hair_Alpha', 'Hair_Alpha', (165.0, 635.0)], 
            ['ShaderNodeBsdfDiffuse', 'Diffuse BSDF', 'Hair_Diffuse', 'Hair_Diffuse', (515.0, 400.0)], 
            ['ShaderNodeBsdfTransparent', 'Transparent BSDF', 'Hair_Transparency', 'Hair_Transparency', (510.0, 500.0338439941406)], 
            ['ShaderNodeMixShader', 'Mix Shader', 'Diffuse_Mix', 'Diffuse_Mix', (750.0, 550.0)], 
            ['ShaderNodeMixShader', 'Mix Shader', 'Color_Mix', 'Color_Mix', (1030.0, 615.0)], 
            ['ShaderNodeOutputMaterial', 'Material Output', 'Material Output', 'Material Output', (1175.0, 280.0)],
            ['ShaderNodeTexImage', 'Image Texture', 'Hair_Displacement', 'Hair_Displacement', (902.8016967773438, 90.0456771850586)]]
        self.hair_card_links = [['nodes["Gradient_Color"].outputs[0]', 'nodes["Tip_Color"].inputs[1]'], 
            ['nodes["Tip_Color"].outputs[0]', 'nodes["Main_Diffuse"].inputs[0]'], 
            ['nodes["Main_Color"].outputs[0]', 'nodes["Gradient_Color"].inputs[1]'], 
            ['nodes["Hair Info"].outputs[1]', 'nodes["Gradient_Control"].inputs[0]'], 
            ['nodes["Hair Info"].outputs[1]', 'nodes["Tip_Control"].inputs[0]'], 
            ['nodes["Hair Info"].outputs[4]', 'nodes["Main_Contrast"].inputs[0]'], 
            ['nodes["Highlight_Mix"].outputs[0]', 'nodes["Highlight_Mix_2"].inputs[0]'], 
            ['nodes["Main_Diffuse"].outputs[0]', 'nodes["Highlight_Mix"].inputs[0]'], 
            ['nodes["Gradient_Control"].outputs[0]', 'nodes["Gradient_Color"].inputs[0]'], 
            ['nodes["Main_Contrast"].outputs[0]', 'nodes["Main_Color"].inputs[0]'], 
            ['nodes["Tip_Control"].outputs[0]', 'nodes["Tip_Color"].inputs[0]'], 
            ['nodes["Main_Highlight"].outputs[0]', 'nodes["Highlight_Mix"].inputs[1]'], 
            ['nodes["Secondary_Highlight"].outputs[0]', 'nodes["Highlight_Mix_2"].inputs[1]'], 
            ['nodes["Highlight_Mix_2"].outputs[0]', 'nodes["Color_Mix"].inputs[2]'], 
            ['nodes["Hair_Alpha"].outputs[0]', 'nodes["Hair_Diffuse"].inputs[0]'], 
            ['nodes["Hair_Alpha"].outputs[1]', 'nodes["Diffuse_Mix"].inputs[0]'], 
            ['nodes["Hair_Alpha"].outputs[1]', 'nodes["Color_Mix"].inputs[0]'], 
            ['nodes["Hair_Diffuse"].outputs[0]', 'nodes["Diffuse_Mix"].inputs[2]'], 
            ['nodes["Hair_Transparency"].outputs[0]', 'nodes["Diffuse_Mix"].inputs[1]'], 
            ['nodes["Diffuse_Mix"].outputs[0]', 'nodes["Color_Mix"].inputs[1]'], 
            ['nodes["Color_Mix"].outputs[0]', 'nodes["Material Output"].inputs[0]'],
            ['nodes["Hair_Displacement"].outputs[0]', 'nodes["Material Output"].inputs[2]']]
        self.hair_card_default = [['Hair_Alpha', self.def_image, (0.6079999804496765, 0.6079999804496765, 0.6079999804496765), 'REPEAT', (0.800000011920929, 0.800000011920929, 0.800000011920929), 0.0, 'MIX', 1.0, 1.0, 1.0, False, 'RGB', 1.0, (0.0, 0.0, 0.0, 1.0), 0.0, 'RGB', 1.0, (1.0, 1.0, 1.0, 1.0), 1.0, 'NEAR', 'LINEAR', 1, 1, 1, 0, False, False],
            ['Hair_Displacement', self.def_disp, (0.6079999804496765, 0.6079999804496765, 0.6079999804496765), 'REPEAT', (0.800000011920929, 0.800000011920929, 0.800000011920929), 0.0, 'MIX', 1.0, 1.0, 1.0, False, 'RGB', 1.0, (0.0, 0.0, 0.0, 1.0), 0.0, 'RGB', 1.0, (1.0, 1.0, 1.0, 1.0), 1.0, 'NEAR', 'LINEAR', 1, 0, 1, -1, False, False],
            ['Gradient_Color', 'MULTIPLY', True, 0.5, (0.466727614402771, 0.3782432973384857, 0.19663149118423462, 1.0), (0.2325773388147354, 0.15663157403469086, 0.07910887151956558, 1.0)],
            ['Tip_Color', 'SCREEN', True, 0.5, (0.466727614402771, 0.3782432973384857, 0.19663149118423462, 1.0), (0.38887712359428406, 0.28217148780822754, 0.10125808417797089, 1.0)],
            ['Main_Color', 'MIX', True, 0.5, (0.466727614402771, 0.3782432973384857, 0.19663149118423462, 1.0), (0.1809358447790146, 0.11345928907394409, 0.04037227854132652, 1.0)],
            ['Main_Diffuse', (0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0), 0.0, (0.0, 0.0, 0.0)], 
            ['Gradient_Control', [[0.045454543083906174, (1.0, 1.0, 1.0, 1.0)], [0.8909090161323547, (0.0, 0.0, 0.0, 1.0)]]], 
            ['Main_Contrast', [[0.08181827515363693, (0.0, 0.0, 0.0, 1.0)], [0.863636314868927, (1.0, 1.0, 1.0, 1.0)]]], 
            ['Tip_Control', [[0.5045454502105713, (0.0, 0.0, 0.0, 1.0)], [1.0, (1.0, 1.0, 1.0, 1.0)]]], 
            ['Main_Highlight', 'GGX', (0.08054633438587189, 0.0542692169547081, 0.030534733086824417, 1.0), 0.25, (0.0, 0.0, 0.0)], 
            ['Secondary_Highlight', 'GGX', (0.023630360141396523, 0.02180372178554535, 0.018096407875418663, 1.0), 0.15000000596046448, (0.0, 0.0, 0.0)]]
    '''
    '''
    def grease_to_curve(self):
        obj = bpy.data.objects
        coll = bpy.context.collection
        cl = bpy.data.collections
        h_acc = get_collection(self.collection)
        h_c = get_subcollection(self.hair_card, self.collection)
        a_co = self.co
        active_ob(self.grease_pencil, None)
        curves = []
        for i, v in enumerate(a_co):
            co = v
            count = len(co)
            n = "{}_C{}".format(self.hair_card, i)
            curves.append(n)
            nc = bpy.data.curves.new(n, type='CURVE')
            cob = bpy.data.objects.new(n, nc)
            curve = obj[n]
            curve.matrix_world = self.object.matrix_world
            h_c.objects.link(curve)
            nc.dimensions = '3D'
            bez = nc.splines.new('BEZIER')
            bz = bez.bezier_points
            bz.add((count - 1))
            bz.foreach_set("co", co.ravel())
            for i, p in enumerate(co):
                bz[i].handle_left_type="AUTO"
                bz[i].handle_right_type="AUTO"
            active_ob(n, None)
            curve.data.extrude = self.width
            bpy.ops.object.editmode_toggle()
            bpy.ops.curve.select_all(action='SELECT')
            bpy.context.object.data.use_uv_as_generated = True
            bpy.context.object.data.use_auto_texspace = True
            bpy.ops.curve.handle_type_set(type='ALIGNED')
            bpy.ops.transform.tilt(value=1.5708, mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
            bpy.ops.object.editmode_toggle()
            material = get_material(self.hair_card)
            material.use_nodes = True
            nodes = material.node_tree.nodes
            clear_node(material)
            for i in self.hair_card_setup:
                add_shader_node(material, i[0], i[2], i[3], i[4])
            for i in self.hair_card_links:
                add_node_link(material, eval(i[0]), eval(i[1]))
            curve.data.materials.append(material)
            args = self.hair_card_default
            for arg in args:
                id = nodes[arg[0]].bl_idname
                fd = func_dict(shader_set_dict(), id)
                data = args[args.index(arg)]
                fd(*data)
            material.blend_method = 'CLIP'
            material.shadow_method = 'OPAQUE'
            material.pass_index = 32

@timeit
def gp_to_curve(ob, dist, width):
    gp_simplify(ob, dist)
    GreaseHairCards(ob, width).grease_to_curve()
    hide_ob(ob, True)

@timeit
def gp_convert_to_mesh(ob):
    obj = bpy.data.objects
    cl = bpy.data.collections
    col =  bpy.context.collection
    h_acc = get_collection("Hair_Accessories")
    h_c = get_subcollection("Hair_GP", "Hair_Accessories")
    add_emp = lambda l: bpy.ops.object.empty_add(type="SPHERE", radius=0.1, align='WORLD', location=l, rotation=(0,0,0))
    hc_name = "{}_Mesh".format(ob)
    active_ob(ob, None)
    bpy.ops.object.convert(target='MESH', keep_original=True)
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.follow_active_quads(mode='LENGTH_AVERAGE')
    bpy.ops.object.editmode_toggle()
    bpy.context.object.name = hc_name
    bpy.context.object.data.name = hc_name
    obj[hc_name].data.materials.append(obj[ob].active_material)
    active_ob(hc_name, None)
    set_gp_uv(hc_name)
    bpy.ops.object.select_all(action='DESELECT')
    if "Hair_Parent" not in [i.name for i in obj]:
        rm90 = rotation_matrix(np.radians(-90), 0, 0)
        rm90_4x4 = Matrix(rm90).to_4x4()
        arm = [i for i in bpy.data.objects if i.type == 'ARMATURE'][0]
        bmt = arm.pose.bones['head'].matrix.copy()
        blc = arm.pose.bones['head'].head
        add_emp(blc)
        bpy.context.object.name = "Hair_Parent"
        h_p = obj["Hair_Parent"]
        h_p.matrix_world = (bmt @ rm90_4x4)
        active_ob("Hair_Parent", None)
        bpy.ops.object.constraint_add(type='COPY_TRANSFORMS')
        bpy.context.object.constraints["Copy Transforms"].target = arm
        bpy.context.object.constraints["Copy Transforms"].subtarget = "head"
        h_acc.objects.link(h_p)
        col.objects.unlink(h_p)
    hide_ob("Hair_Parent", False)
    active_ob("Hair_Parent", [hc_name])
    bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
    hide_ob(ob, True)
    bpy.ops.object.select_all(action='DESELECT')
    active_ob(hc_name, None)

@timeit
def gp_armature(ob):
    ccoll = bpy.context.collection
    h_acc = get_collection("Hair_Accessories")
    tar = "{}_Targets".format(ob)
    h_tar = get_subcollection(tar, "Hair_Accessories")
    add_emp = lambda l, r: bpy.ops.object.empty_add(type="SPHERE", radius=0.1, align='WORLD', location=l, rotation=r)
    e_name = "{}_Edge".format(ob)
    m_name = "{}_Mesh".format(ob)
    a_name = "{}_Armature".format(ob)
    obj = bpy.data.objects
    curve = obj[ob]
    width = curve.data.extrude
    # gp_convert_to_mesh(ob)
    active_ob(ob, None)
    curve.data.extrude = 0
    bpy.ops.object.convert(target='MESH', keep_original=True)
    bpy.context.object.name = e_name
    bpy.context.object.data.name = e_name
    curve.data.extrude = width
    ed = obj[e_name]
    active_ob(e_name, None)
    vt = ed.data.vertices
    count = len(vt)
    eco = np.empty(count * 3)
    vt.foreach_get("co", eco)
    eco.shape = (count, 3)
    vg = ed.vertex_groups
    nvg = vg.new(name="pin")
    index = np.arange(count).ravel().tolist()
    nvg.add(index, 1.0, 'REPLACE')
    wts = np.linspace(1.0, .1, num=count).tolist()
    for i, w in enumerate(wts):
        vt[i].groups[0].weight = w
    emod = ed.modifiers.new("Cloth", 'CLOTH')
    ed.modifiers['Cloth'].settings.vertex_group_mass = "pin"
    add_armature(a_name, h_acc.name)
    active_ob(a_name, None)
    eidx = edge_idx_chain(count)
    bones = []
    for i, v in enumerate(eidx):
        n = "{}_B{}".format(ob, i)
        add_bone(a_name, n, eco[v])
        bones.append(n)
    bones.reverse()
    arm = obj[a_name]
    pb = arm.pose.bones
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
    eb = arm.data.edit_bones
    for idx, b in enumerate(bones[:-1]):
        ix = idx + 1
        eb[b].parent = eb[bones[ix]]
        eb[b].use_connect = True
    bpy.ops.object.mode_set(mode='OBJECT')
    targets = []
    bones.reverse()
    for i, v in enumerate(eco[:-1]):
        r = pb[bones[i]].matrix.to_euler('XYZ')
        nm = "{}_Target_{}".format(ob, i)
        add_emp(v, r)
        obj["Empty"].name = nm
        h_tar.objects.link(obj[nm])
        ccoll.objects.unlink(obj[nm])
        targets.append(nm)
    tidx = count - 1
    nml = "{}_Target_{}".format(ob, tidx)
    rl = pb[bones[-1]].matrix.to_euler('XYZ')
    add_emp(eco[tidx], rl)
    obj["Empty"].name = nml
    h_tar.objects.link(obj[nml])
    ccoll.objects.unlink(obj[nml])
    targets.append(nml)
    for ix, k in enumerate(targets):
        active_ob(e_name, [k])
        bpy.ops.object.mode_set(mode='EDIT', toggle=False)
        par = bpy.data.objects[e_name]
        ch = bpy.data.objects[k]
        ch.parent = par
        ch.matrix_world = par.matrix_world
        ch.parent_type = 'VERTEX'
        ch.parent_vertices[0] = ix
        bpy.ops.object.mode_set(mode='OBJECT')
    active_ob(a_name, None)
    for b, t in list(zip(bones, targets[1:])):
        bct = pb[b].constraints.new('DAMPED_TRACK')
        bct.target = obj[t]
    bpy.ops.object.select_all(action='DESELECT')
    curve.data.extrude = width
    active_ob(a_name, [m_name])
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    bpy.ops.object.select_all(action='DESELECT')
    if "Hair_Parent" not in [i.name for i in obj]:
        rm90 = rotation_matrix(np.radians(-90), 0, 0)
        rm90_4x4 = Matrix(rm90).to_4x4()
        arm = [i for i in bpy.data.objects if i.type == 'ARMATURE'][0]
        bmt = arm.pose.bones['head'].matrix.copy()
        blc = arm.pose.bones['head'].head
        add_emp(blc, (0,0,0))
        bpy.context.object.name = "Hair_Parent"
        h_p = obj["Hair_Parent"]
        h_p.matrix_world = (bmt @ rm90_4x4)
        active_ob("Hair_Parent", None)
        bpy.ops.object.constraint_add(type='COPY_TRANSFORMS')
        bpy.context.object.constraints["Copy Transforms"].target = arm
        bpy.context.object.constraints["Copy Transforms"].subtarget = "head"
        h_acc.objects.link(h_p)
        col.objects.unlink(h_p)
    hide_ob("Hair_Parent", False)
    active_ob("Hair_Parent", [e_name, a_name])
    bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
    bpy.ops.object.select_all(action='DESELECT')
    hide_ob(e_name, True)
    hide_ob(a_name, True)
    hide_ob("Hair_Parent", True)
    for t in targets:
        hide_ob(t, True)
    hide_ob(ob, True)
    active_ob(m_name, None)

#---------------------------------------------------------

def get_uv_co(ob):
    obj = bpy.data.objects
    o = obj[ob]
    data = o.data.uv_layers.active.data
    count = len(data)
    co = np.empty(count * 2)
    data.foreach_get('uv', co)
    co.shape = (count, 2)
    return co

@timeit
def set_hc_uv(ob):
    obj = bpy.data.objects
    o = obj[ob]
    co = get_uv_co(ob)
    bot = np.argwhere(co[:,1] == 0)
    top = np.argwhere(co[:,1] == 1)
    bct = len(bot)
    tct = len(top)
    bloc = np.linspace(0,1, num=bct)
    tloc = np.linspace(0,1, num=tct)
    for i, b in enumerate(bot.ravel()):
        bx = (o.data.uv_layers.active.data[b].uv.x * bloc[1] * 2.5)
        o.data.uv_layers.active.data[b].uv = (bx, 0.0)
    for j, t in enumerate(top.ravel()):
        tx = (o.data.uv_layers.active.data[t].uv.x * tloc[1] * 2.5)
        o.data.uv_layers.active.data[t].uv = (tx, 1.0)

@timeit
def set_gp_uv(ob):
    obj = bpy.data.objects
    o = obj[ob]
    co = get_uv_co(ob)
    bot = np.argwhere(co[:,1] == 0)
    top = np.argwhere(co[:,1] == 1)
    bct = len(bot)
    tct = len(top)
    bloc = np.linspace(0,1, num=bct)
    tloc = np.linspace(0,1, num=tct)
    for i, b in enumerate(bot.ravel()):
        bx = (o.data.uv_layers.active.data[b].uv.x + .2)
        o.data.uv_layers.active.data[b].uv = (bx, 0.0)
    for j, t in enumerate(top.ravel()):
        tx = (o.data.uv_layers.active.data[t].uv.x + .2)
        o.data.uv_layers.active.data[t].uv = (tx, 1.0)


#---------------------------------------------------------

@timeit
def rig_influence(val, bones):
    obj = bpy.data.objects
    pb = obj["rig"].pose.bones
    for bone in bones:
        pb[bone]["IK_FK"] = val

@timeit
def CT_influence(val, bones):
    obj = bpy.data.objects
    pb = obj["rig"].pose.bones
    for bone in bones:
        pb[bone].constraints["Copy Transforms"].influence = val

#---------------------------------------------------------

@timeit
def obj_driver(Obj, path, idx, tar, trans, exp='var'):
    obj = bpy.data.objects
    dr = obj[Obj].driver_add(path, idx).driver
    dr.type = 'SCRIPTED'
    dr.expression = exp
    var = dr.variables.new()
    var.type = 'TRANSFORMS'
    var.targets[0].id = obj[tar]
    var.targets[0].transform_type = trans
    var.targets[0].transform_space = 'WORLD_SPACE'
    #var.targets[0].bone_target = bt

@timeit
def bone_driver(Bone, path, idx, tar, trans):
    obj = bpy.data.objects
    dr = obj["rig"].pose.bones[Bone].driver_add(path, idx).driver
    dr.type = 'SCRIPTED'
    dr.expression = 'var'
    var = dr.variables.new()
    var.type = 'TRANSFORMS'
    var.targets[0].id = obj[tar]
    var.targets[0].transform_type = trans
    var.targets[0].transform_space = 'WORLD_SPACE'

@timeit
def rename_vgroup(vg, newName):
    rem_vg = lambda g: vg.remove(vg.get(g))
    rem_vg("DEF-spine.004")
    for new, old in newName:
        o = vg.get(old)
        new = 'DEF-{}'.format(new)
        n = vg.get(new)
        if (o is None) or (n is None):
            continue
        rem_vg(new)
        if 'pelvis' in new:
            new = 'DEF-spine'
        o.name = new

#---------------------------------------------------------

@timeit
def align_rig(fingers=True):
    obj = bpy.data.objects
    # MB_rig = bpy.context.scene.mblab_fitref_name
    # mba = obj[MB_rig].parent.name
    try:
        enable_addon(["rigify"])
    except:
        pass
    if fingers == True:
        bpy.ops.object.armature_human_metarig_add()
    else:
        bpy.ops.object.armature_basic_human_metarig_add()
    p_extra = ["pelvis.L", "pelvis.R"]
    bcmb = bone_collector(0)
    bcmr = bone_collector('metarig')
    arm_MR = bpy.data.objects['metarig']
    spine = bcmr['spine']
    pelvis = bcmb['pelvis']
    pscale = (spine[4] / pelvis[4])
    if not arm_MR.data.is_editmode:
        bpy.ops.object.editmode_toggle()
    for i in list(zip(MR_list, MB_list)):
        MR = i[0]
        MB = i[1]
        arm_MR.data.edit_bones[MR].head = bcmb[MB][0]
        arm_MR.data.edit_bones[MR].tail = bcmb[MB][1]
        arm_MR.data.edit_bones[MR].head_radius = bcmb[MB][2]
        arm_MR.data.edit_bones[MR].tail_radius = bcmb[MB][3]
    for p in p_extra:
        vec = bcmr[p][1] - bcmr[p][0]
        arm_MR.data.edit_bones[p].head = pelvis[0]
        arm_MR.data.edit_bones[p].tail = pelvis[0] + (vec * pscale)
        arm_MR.data.edit_bones[p].head_radius = pelvis[2]
        arm_MR.data.edit_bones[p].tail_radius = pelvis[3]
    arm_MR.data.edit_bones['spine.004'].head = bcmb['spine03'][1]
    arm_MR.data.edit_bones['spine.004'].tail = bcmb['neck'][0]
    arm_MR.data.edit_bones[p_extra[0]].tail.x = bcmb['thigh_L'][0][0]
    arm_MR.data.edit_bones[p_extra[0]].tail.y = bcmb['thigh_L'][0][1]
    arm_MR.data.edit_bones[p_extra[1]].tail.x = bcmb['thigh_R'][0][0]
    arm_MR.data.edit_bones[p_extra[1]].tail.y = bcmb['thigh_R'][0][1]
    hdL = bcmr["heel.02.L"][0] + (bcmb["foot_L"][0] - bcmr["foot.L"][0])
    hdL[2] = 0.0
    tlL = hdL + np.array([bcmr["heel.02.L"][4], 0, 0])
    hdR = bcmr["heel.02.R"][0] + (bcmb["foot_R"][0] - bcmr["foot.R"][0])
    hdR[2] = 0.0
    tlR = hdR + np.array([-bcmr["heel.02.R"][4], 0, 0])
    arm_MR.data.edit_bones["heel.02.L"].head = hdL
    arm_MR.data.edit_bones["heel.02.L"].tail = tlL
    arm_MR.data.edit_bones["heel.02.R"].head = hdR
    arm_MR.data.edit_bones["heel.02.R"].tail = tlR
    if fingers == True:
        bpy.ops.armature.select_all(action='DESELECT')
        for bo in MR_face:
            arm_MR.data.edit_bones[bo].select = True
        bpy.ops.armature.delete()
        bpy.ops.armature.select_all(action='DESELECT')
        for i in list(zip(MR_fingers, MB_fingers)):
            MR = i[0]
            MB = i[1]
            arm_MR.data.edit_bones[MR].head = bcmb[MB][0]
            arm_MR.data.edit_bones[MR].tail = bcmb[MB][1]
            arm_MR.data.edit_bones[MR].head_radius = bcmb[MB][2]
            arm_MR.data.edit_bones[MR].tail_radius = bcmb[MB][3]
    if arm_MR.data.is_editmode:
        bpy.ops.object.editmode_toggle()


def limb_physics():
    coll = bpy.context.collection
    obj = bpy.data.objects
    bc = bone_collector('metarig')
    armL = ["upper_arm.L", "hand.L"]
    armR = ["upper_arm.R", "hand.R"]
    legL = ["thigh.L", "foot.L"]
    legR = ["thigh.R", "foot.R"]
    parts = ["armL", "armR", "legL", "legR"]
    ed = lambda b: [bc[b[0]][0], bc[b[0]][1], bc[b[1]][0]]
    edr = lambda b: [bc[b[1]][0], bc[b[0]][1], bc[b[0]][0]]
    coaL = ed(armL)
    coaR = ed(armR)
    colL = edr(legL)
    colR = edr(legR)
    coors = [coaL, coaR, colL, colR]
    # vt = obj[object].data.vertices
    # vg = obj[object].vertex_groups
    v_group = "pin"
    pkcoll = "Pino_Kio_Rig"
    stiffness = 0.25
    scount = 3
    sc = np.arange(scount)
    sc1 = np.arange((scount+1))
    ed = list(zip(sc1[:-2], sc1[1:-1]))
    wts = [1, .5, .25] #np.linspace(stiffness, 1, num=scount).tolist()
    # wts.reverse()
    add_emp = lambda l: bpy.ops.object.empty_add(type='PLAIN_AXES', radius=.005, align='WORLD', location=l, rotation=(0,0,0))
    add_edge = lambda Name, co: obj_new(Name, [Vector(i) for i in co], [[0,1], [1, 2]], [], pkcoll)
    #index = sc.tolist()
    p_list = []
    for p, c in list(zip(parts, coors)):
        add_edge(p, c)
    # for i, v in enumerate(coors):
    #     pt = parts[i]
    #     emp_names = []
    #     for ix, ve in v:
    #         prt = "{}_{}".format(pt, ix)
    #         add_emp(ve)
    #         obj["Empty"].name = prt
    #         emp_names.append(prt)
    #     p_dict.append([pt, emp_names])
    # print(p_list)
    # for o in p_list:
    #     k, v = o[0], o[1]
    #     active_ob(k, v)
    #     for i, j in enumerate(v):
    #         adoption(k, j, 'VERTEX', i)

@timeit
def parent_rig():
    obj = bpy.data.objects
    MB_rig = bpy.context.scene.mblab_fitref_name
    arm_MB = obj[MB_rig].parent #bpy.data.objects[[i.name for i in bpy.data.objects if i.type == 'ARMATURE'][0]]
    body = obj[MB_rig]
    arm_rig = bpy.data.objects['rig']
    arm_MR = bpy.data.objects['metarig']
    vg = body.vertex_groups
    comp = np.array(list(zip(MR_parts, MB_parts)))
    bpy.ops.object.select_all(action='DESELECT')
    active_ob(arm_rig.name, [body.name])
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    rename_vgroup(vg, comp)
    arm_MB.hide_viewport = True
    arm_MR.hide_viewport = True
    arm_rig.show_in_front = True
    show_rig(arm_rig, (tweak_bones + fk_bones), False)
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.posemode_toggle()
    lock_rig_loc()
    rig_stretch_locks()
    bpy.ops.object.posemode_toggle()
    #limb_physics()


@timeit
def pino_kio_rig():
    coll = bpy.context.collection
    pkcoll = get_collection("Pino_Kio_Rig")
    bcmb = bone_collector(0)
    bcrig = bone_collector(-1)
    pkrig = ["torso", "hand_ik.L", "hand_ik.R", "foot_ik.L", "foot_ik.R"]
    e_type = ['CUBE', 'SPHERE', 'SPHERE', 'CIRCLE', 'CIRCLE']
    obj = bpy.data.objects
    en = 'Empty'
    pb = obj['rig'].pose.bones
    add_emp = lambda t, rad, l, r: bpy.ops.object.empty_add(type=t, radius=rad, align='WORLD', location=l, rotation=r)
    edge_maker = lambda Name, co: obj_new(Name, co, [[0, 1, 2]], [], pkcoll.name)
    ld = lambda ob: limit_distance(ob, "torso", 'LIMITDIST_INSIDE')
    ct = lambda Bone: bone_copy_transform('rig', Bone, Bone)
    ctb = lambda ob, ht: copy_transform(ob, 'rig', ob, ht)
    cr = lambda Bone, tar, i: bone_copy_rot("rig", Bone, tar, i)
    for r in list(zip(e_type, pkrig)):
        if r[1] == "torso":
            rad = 0.15
        else:
            rad = 0.1
        if r[1] == pkrig[1]:
            rot = (0,0,np.radians(-90))
        elif r[1] == pkrig[2]:
            rot = (0,0,np.radians(90))
        else:
            rot = (0,0,0)
        add_emp(r[0], rad, bcrig[r[1]][0], rot)
        obj[en].name = r[1]
        pkcoll.objects.link(obj[r[1]])
        coll.objects.unlink(obj[r[1]])
    add_emp('SPHERE', 0.2, bcrig['chest'][0], (0, 0, 0))
    obj[en].name = "chest_control"
    pkcoll.objects.link(obj["chest_control"])
    coll.objects.unlink(obj["chest_control"])
    add_emp('SPHERE', 0.2, bcrig['torso'][0], (0, 0, 0))
    obj[en].name = "hip_control"
    pkcoll.objects.link(obj["hip_control"])
    coll.objects.unlink(obj["hip_control"])
    add_emp('SPHERE', 0.2, bcrig['head'][0], (np.radians(90), 0, 0))
    obj[en].name = "head_control"
    pkcoll.objects.link(obj["head_control"])
    coll.objects.unlink(obj["head_control"])
    add_emp('SPHERE', 0.05, bcrig['upper_arm_ik.L'][0], (0, 0, 0))
    obj[en].name = 'upper_arm_ik.L'
    pkcoll.objects.link(obj['upper_arm_ik.L'])
    coll.objects.unlink(obj['upper_arm_ik.L'])
    add_emp('SPHERE', 0.05, bcrig['upper_arm_ik.L'][0], (0, 0, 0))
    obj[en].name = 'upper_arm_ik.R'
    pkcoll.objects.link(obj['upper_arm_ik.R'])
    coll.objects.unlink(obj['upper_arm_ik.R'])
    add_emp('SPHERE', 0.05, bcrig['thigh_ik.L'][0], (0, 0, 0))
    obj[en].name = 'thigh_ik.L'
    pkcoll.objects.link(obj['thigh_ik.L'])
    coll.objects.unlink(obj['thigh_ik.L'])
    add_emp('SPHERE', 0.05, bcrig['thigh_ik.R'][0], (0, 0, 0))
    obj[en].name = 'thigh_ik.R'
    pkcoll.objects.link(obj['thigh_ik.R'])
    coll.objects.unlink(obj['thigh_ik.R'])
    #
    add_emp('SPHERE', 0.05, bcrig['upper_arm_ik.L'][0], (0, 0, 0))
    obj[en].name = 'shoulder_socket.L'
    pkcoll.objects.link(obj['shoulder_socket.L'])
    coll.objects.unlink(obj['shoulder_socket.L'])
    add_emp('SPHERE', 0.05, bcrig['upper_arm_ik.R'][0], (0, 0, 0))
    obj[en].name = 'shoulder_socket.R'
    pkcoll.objects.link(obj['shoulder_socket.R'])
    coll.objects.unlink(obj['shoulder_socket.R'])
    #
    active_ob("chest_control", ["head_control"])
    bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
    active_ob("torso", ["chest_control"])
    bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
    for pk in pkrig[3:]:
        ld(pk)
    limit_distance("hand_ik.L", 'upper_arm_ik.L', 'LIMITDIST_INSIDE')
    limit_distance("hand_ik.R", 'upper_arm_ik.R', 'LIMITDIST_INSIDE')
    limit_distance('shoulder_socket.L', 'chest_control', 'LIMITDIST_OUTSIDE')
    limit_distance('shoulder_socket.R', 'chest_control', 'LIMITDIST_OUTSIDE')
    #
    limit_bone_distance("hand_ik.L", "forearm_tweak.L.001", 'LIMITDIST_INSIDE')
    limit_bone_distance("forearm_tweak.L.001", "forearm_tweak.L", 'LIMITDIST_INSIDE')
    limit_bone_distance("forearm_tweak.L", "upper_arm_tweak.L.001", 'LIMITDIST_INSIDE')
    limit_bone_distance("upper_arm_tweak.L.001", "upper_arm_tweak.L", 'LIMITDIST_INSIDE')
    limit_bone_distance("hand_ik.R", "forearm_tweak.R.001", 'LIMITDIST_INSIDE')
    limit_bone_distance("forearm_tweak.R.001", "forearm_tweak.R", 'LIMITDIST_INSIDE')
    limit_bone_distance("forearm_tweak.R", "upper_arm_tweak.R.001", 'LIMITDIST_INSIDE')
    limit_bone_distance("upper_arm_tweak.R.001", "upper_arm_tweak.R", 'LIMITDIST_INSIDE')
    #
    copy_loc("hip_control", "torso")
    obj_driver("hip_control", 'rotation_euler', 2, "torso", 'ROT_Z', 'var * 1.5')
    #
    bpy.ops.object.posemode_toggle()
    bone_damp_trac("shoulder.L", "upper_arm_ik.L", 'TRACK_Y')
    bone_damp_trac("shoulder.R", "upper_arm_ik.R", 'TRACK_Y')
    cr("head", "head_control", 1)
    cr("neck", "head_control", 0.5)
    cr("chest", "chest_control", 1)
    cr("hips", "hip_control", 1)
    cr("foot_heel_ik.L", "foot_ik.L", 1)
    pb["foot_heel_ik.L"].constraints["Copy Rotation"].target_space = 'LOCAL'
    cr("foot_heel_ik.R", "foot_ik.R", 1)
    pb["foot_heel_ik.R"].constraints["Copy Rotation"].target_space = 'LOCAL'
    obj_limit_rot("head_control", [45, 145, -45, 45, -80, 80], 'LOCAL')
    obj_limit_rot("hip_control", [-45, 45, -15, 15, -45, 45], 'WORLD')
    obj_limit_rot("chest_control", [-60, 60, -25, 25, -35, 35], 'WORLD')
    obj_limit_rot("foot_ik.L", [-15, 30, -25, 15, -25, 25], 'WORLD')
    obj_limit_rot("foot_ik.R", [-15, 30, -15, 25, -25, 25], 'WORLD')
    bone_limit_rot("shoulder.L", [-10, 10, 0, 0, -90, -45], 'POSE')
    bone_limit_rot("shoulder.R", [-10, 10, 0, 0, 45, 90], 'POSE')
    bone_driver("foot_heel_ik.L", 'rotation_euler', 0, "foot_ik.L", 'ROT_X')
    bone_driver("foot_heel_ik.L", 'rotation_euler', 0, "foot_ik.L", 'ROT_X')
    bone_driver("thigh_ik.L", 'rotation_euler', 2, "foot_ik.L", 'ROT_Z')
    bone_driver("thigh_ik.R", 'rotation_euler', 2, "foot_ik.R", 'ROT_Z')
    for rb in pkrig:
        ct(rb)
    ctb("upper_arm_ik.L", 1)
    ctb("upper_arm_ik.R", 1)
    ctb("thigh_ik.L", 1)
    ctb("thigh_ik.R", 1)
    copy_transform('shoulder_socket.L', 'rig', 'upper_arm_ik.L', 0)
    copy_transform('shoulder_socket.R', 'rig', 'upper_arm_ik.R', 0)
    bpy.ops.object.posemode_toggle()
    lock_pk_rig_loc()
    show_rig(bpy.data.objects["rig"], (rig_bones + ik_bones), False)
    show_rig(bpy.data.objects["rig"], (root_b), True)

@timeit
def MB_to_Rigify():
    import rigify
    align_rig()
    rigify.ui.generate.generate_rig(bpy.context, bpy.data.objects['metarig'])
    parent_rig()

