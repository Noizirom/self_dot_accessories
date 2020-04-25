# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 3
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

bl_info = {
    "name": "Self_Dot_Accessories",
    "author": "Noizirom",
    "version": (1, 0, 1),
    "blender": (2, 80, 0),
    "location": "View3D > Tools > Self_Dot_Accessories",
    "description": "Adds Accessories to Characters",
    "warning": "",
    "wiki_url": "",
    "category": "Characters",
}

import bpy
from bpy.types import Operator
from bpy.props import FloatVectorProperty
from bpy_extras.object_utils import AddObjectHelper, object_data_add
from mathutils import Vector
from . import self_dot_accessories as SDA

UN_shader_remove = []

def hair_style_list(self, context):
    scn = bpy.context.scene
    gud = SDA.get_universal_dict(SDA.get_filename(SDA.get_hair_dir(), "universal_hair_shader.npz"))
    items = list(gud.keys())
    #[(identifier, name, description, icon, number)
    return [(i, i, i) for i in items]

#Hair color Drop Down List
bpy.types.Scene.acc_hair_color = bpy.props.EnumProperty(
    items=hair_style_list,
    name="Color Select",
    update=hair_style_list)

#Hair new color name
bpy.types.Scene.acc_new_hair_color = bpy.props.StringProperty(
    name="Save Hair Color",
    default="",
    description="Enter name for new hair color")

class OBJECT_OT_Rigify(Operator, AddObjectHelper):
    """Create a Rigify Rig"""
    bl_idname = "acc.rigify"
    bl_label = "Add Rigify Rig"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        SDA.MB_to_Rigify()
        return {'FINISHED'}


class OBJECT_OT_PK_Rig(Operator, AddObjectHelper):
    """Create a Pino Kio Rig"""
    bl_idname = "acc.pk_rig"
    bl_label = "Add Pino Kio Rig"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        SDA.pino_kio_rig()
        return {'FINISHED'}

#####

class OBJECT_OT_head_hair(bpy.types.Operator):
    """Add Hair to Character from Selected Polygons"""
    bl_idname = "acc.head_hair"
    bl_label = "Hair from Selected"
    bl_options = {'REGISTER', 'INTERNAL', 'UNDO'}

    def execute(self, context):
        SDA.hair_from_selected()
        return {'FINISHED'}

class OBJECT_OT_add_shader(bpy.types.Operator):
    """Add Shader to Hair"""
    bl_idname = "acc.default_shader"
    bl_label = "Add Shader to Hair"
    bl_options = {'REGISTER', 'INTERNAL', 'UNDO'}

    def execute(self, context):
        SDA.add_hair_shader()
        return {'FINISHED'}

class OBJECT_OT_convert_to_curve(bpy.types.Operator):
    """Convert Particle Hair to Hair Cards"""
    bl_idname = "acc.hair_card_clump"
    bl_label = "Hair Cards from Particle Hair"
    bl_options = {'REGISTER', 'INTERNAL', 'UNDO'}

    def execute(self, context):
        SDA.convert_to_curve(bpy.context.object.name)
        return {'FINISHED'}

class OBJECT_OT_convert_to_mesh(bpy.types.Operator):
    """Convert Hair Cards to Hair Mesh"""
    bl_idname = "acc.hair_card_mesh"
    bl_label = "Hair Mesh from Hair Cards"
    bl_options = {'REGISTER', 'INTERNAL', 'UNDO'}

    def execute(self, context):
        SDA.convert_to_mesh(bpy.context.object.name)
        return {'FINISHED'}

class OBJECT_OT_seperate_cards(bpy.types.Operator):
    """Seperate Hair Card Clump"""
    bl_idname = "acc.seperate_card"
    bl_label = "Seperate Hair Cards"
    bl_options = {'REGISTER', 'INTERNAL', 'UNDO'}

    def execute(self, context):
        SDA.seperate_cards(bpy.context.object.name)
        return {'FINISHED'}

class OBJECT_OT_curve_armature(bpy.types.Operator):
    """Convert Hair Card to Physic Mesh"""
    bl_idname = "acc.curve_arm"
    bl_label = "Physics Responsive Hair"
    bl_options = {'REGISTER', 'INTERNAL', 'UNDO'}

    def execute(self, context):
        SDA.curve_armature(bpy.context.object.name)
        return {'FINISHED'}

class OBJECT_OT_grease_hair(bpy.types.Operator):
    """Add Grease Pencil Hair to Character"""
    bl_idname = "acc.gp_hair"
    bl_label = "Add Grease Pencil"
    bl_options = {'REGISTER', 'INTERNAL', 'UNDO'}

    def execute(self, context):
        SDA.add_gp()
        return {'FINISHED'}

class OBJECT_OT_convert_grease_to_curve(bpy.types.Operator):
    """Convert Grease Pencil Hair to Hair Cards"""
    bl_idname = "acc.grease_hair_card"
    bl_label = "Hair Cards from Grease Pencil"
    bl_options = {'REGISTER', 'INTERNAL', 'UNDO'}

    def execute(self, context):
        SDA.gp_to_curve(bpy.context.object.name, .1, .01)
        return {'FINISHED'}

class OBJECT_OT_edit_weights(bpy.types.Operator):
    """Edit Weights for Selected Hair Mesh"""
    bl_idname = "acc.edit_hair_weights"
    bl_label = "Edit Physics Hair Weights"
    bl_options = {'REGISTER', 'INTERNAL', 'UNDO'}

    def execute(self, context):
        SDA.edit_weights()
        return {'FINISHED'}

class OBJECT_OT_finish_weights(bpy.types.Operator):
    """Finish Weights for Selected Hair Mesh"""
    bl_idname = "acc.finish_hair_weights"
    bl_label = "Finish Physics Hair Weights"
    bl_options = {'REGISTER', 'INTERNAL', 'UNDO'}

    def execute(self, context):
        SDA.finish_weights()
        return {'FINISHED'}

class OBJECT_OT_vg_splitter(bpy.types.Operator):
    """Add Vertex Groups Splitter to Selected Object"""
    bl_idname = "acc.hair_splitter"
    bl_label = "Add Splitter Object"
    bl_options = {'REGISTER', 'INTERNAL', 'UNDO'}

    def execute(self, context):
        SDA.hair_splitter_target(bpy.context.object.name)
        return {'FINISHED'}

class OBJECT_OT_add_split_vg(bpy.types.Operator):
    """Add Split Vertex Groups to Selected Object"""
    bl_idname = "acc.hair_vg"
    bl_label = "Add Vertex Groups"
    bl_options = {'REGISTER', 'INTERNAL', 'UNDO'}

    def execute(self, context):
        SDA.add_hair_vg(bpy.context.object.name)
        return {'FINISHED'}

class OBJECT_OT_change_hair_color(bpy.types.Operator):
    """Change Selected Hair Color"""
    bl_idname = "acc.change_hair"
    bl_label = "Change Color"
    bl_options = {'REGISTER', 'INTERNAL', 'UNDO'}

    def execute(self, context):
        ob = bpy.context.object
        scn = bpy.context.scene
        style = scn.acc_hair_color
        material = ob.active_material
        nodes = material.node_tree.nodes
        SDA.set_universal_shader(material.name, SDA.get_filename(SDA.get_hair_dir(), "universal_hair_shader.npz"), style)
        return {'FINISHED'}

class OBJECT_OT_add_color_preset(bpy.types.Operator):
    """Add Hair Color to Presets"""
    bl_idname = "acc.add_hair_preset"
    bl_label = "Add Preset"
    bl_options = {'REGISTER', 'INTERNAL', 'UNDO'}

    def execute(self, context):
        scn = bpy.context.scene
        newColor = scn.acc_new_hair_color
        material = bpy.context.object.active_material
        nodes = material.node_tree.nodes
        fileName = SDA.get_filename(SDA.get_hair_dir(), "universal_hair_shader.npz")
        SDA.save_universal_presets(fileName, newColor, SDA.get_all_shader_(nodes))
        return {'FINISHED'}

class OBJECT_OT_remove_color_preset(bpy.types.Operator):
    """Remove Hair Color from Presets"""
    bl_idname = "acc.del_hair_preset"
    bl_label = "Delete Preset"
    bl_options = {'REGISTER', 'INTERNAL', 'UNDO'}

    def execute(self, context):
        scn = bpy.context.scene
        style = scn.acc_hair_color
        global UN_shader_remove
        fileName = SDA.get_filename(SDA.get_hair_dir(), "universal_hair_shader.npz")
        SDA.remove_universal_presets(fileName, style, UN_shader_remove)
        return {'FINISHED'}

#Undo Delete Preset
class OBJECT_OT_undo_remove_color(bpy.types.Operator):
    """Replace Removed Hair Color"""
    bl_idname = "acc.rep_hair_preset"
    bl_label = "Undo Delete Preset"
    bl_options = {'REGISTER', 'INTERNAL', 'UNDO'}

    def execute(self, context):
        scn = bpy.context.scene
        style = scn.acc_hair_color
        global UN_shader_remove
        fileName = SDA.get_filename(SDA.get_hair_dir(), "universal_hair_shader.npz")
        SDA.replace_removed_shader(fileName, UN_shader_remove)
        return {'FINISHED'}



####

class VIEW3D_PT_Dot_Accessories(bpy.types.Panel):
    bl_label = "SELF_DOT_ACCESSORIES"
    bl_idname = "OBJECT_PT_accessories"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_context = 'objectmode'
    bl_category = "Self_Dot_Accessories"

    @classmethod
    def poll(cls, context):
        return context.mode in {'OBJECT', 'EDIT_MESH', 'POSE'}

    def draw(self, context):
        scn = bpy.context.scene
        hair_info = self.layout.box()
        hair_info.label(text="Hair Engine")
        self.layout.operator("acc.head_hair", text="Add Head Hair")
        self.layout.operator("acc.default_shader", text="Add Hair Shader")
        self.layout.prop(scn, 'acc_hair_color')
        self.layout.operator("acc.change_hair", text="Change Hair Color")
        self.layout.prop(scn, 'acc_new_hair_color')
        self.layout.operator("acc.add_hair_preset", text="Add Preset")
        self.layout.operator("acc.del_hair_preset", text="Delete Preset")
        self.layout.operator("acc.rep_hair_preset", text="Undo Delete Preset")
        self.layout.label(text="Add Vert Group")
        self.layout.operator("acc.hair_splitter", text="Add Vertex Group Helper")
        self.layout.operator("acc.hair_vg", text="Add Vertex Groups")
        self.layout.label(text="Hair Cards")
        self.layout.operator("acc.hair_card_clump", text="Convert to Cards")
        self.layout.operator("acc.seperate_card", text="Seperate Hair Cards")
        self.layout.label(text="Grease Pencil Hair Cards")
        self.layout.operator("acc.gp_hair", text="Add Grease Hair")
        self.layout.operator("acc.grease_hair_card", text="Convert to Cards")
        self.layout.label(text="Convert Cards")
        self.layout.operator("acc.hair_card_mesh", text="Convert to Mesh")
        self.layout.operator("acc.curve_arm", text="Convert to Physics")
        self.layout.label(text="Edit Physics Weights")
        self.layout.operator("acc.edit_hair_weights", text="Edit Physics Weights")
        self.layout.operator("acc.finish_hair_weights", text="Finish Physics Weights")
        rig_info = self.layout.box()
        rig_info.label(text="Pino Kio Engine")
        self.layout.operator("acc.rigify", text="MB to Rigify")
        self.layout.operator("acc.pk_rig", text="Add Pino_Kio Rig")


# Registration


# This allows you to right click on a button and link to documentation
def add_object_manual_map():
    url_manual_prefix = "https://docs.blender.org/manual/en/latest/"
    url_manual_mapping = (
        ("bpy.ops.mesh.add_object", "scene_layout/object/types.html"),
    )
    return url_manual_prefix, url_manual_mapping

classes = [
    OBJECT_OT_Rigify,
    OBJECT_OT_PK_Rig,
    OBJECT_OT_head_hair,
    OBJECT_OT_add_shader,
    OBJECT_OT_convert_to_curve,
    OBJECT_OT_convert_to_mesh,
    OBJECT_OT_seperate_cards,
    OBJECT_OT_curve_armature,
    OBJECT_OT_grease_hair,
    OBJECT_OT_convert_grease_to_curve,
    OBJECT_OT_edit_weights,
    OBJECT_OT_finish_weights,
    OBJECT_OT_vg_splitter,
    OBJECT_OT_add_split_vg,
    OBJECT_OT_change_hair_color,
    OBJECT_OT_add_color_preset,
    OBJECT_OT_remove_color_preset,
    OBJECT_OT_undo_remove_color,
    VIEW3D_PT_Dot_Accessories,
]


def register():
    #bpy.utils.register_class(OBJECT_OT_Rigify)
    bpy.utils.register_manual_map(add_object_manual_map)
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    #bpy.utils.unregister_class(OBJECT_OT_Rigify)
    bpy.utils.unregister_manual_map(add_object_manual_map)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
