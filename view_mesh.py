#!/usr/bin/env python3
import sys
import trimesh
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        # Default sample
        mesh_path = Path("C:/Users/Student3/Desktop/ccbd_dogvton/barc_3d_outputs/golden/3d_meshes/golden_golden_retriever_dog_show_bing_00000_3d_mesh.obj")
    else:
        mesh_path = Path(sys.argv[1])
        
    if not mesh_path.exists():
        print(f"File not found: {mesh_path}")
        return
        
    print(f"[*] Loading 3D Mesh: {mesh_path.name}")
    mesh = trimesh.load(str(mesh_path), process=False)
    print(f"  • Vertices: {len(mesh.vertices):,}")
    print(f"  • Faces:    {len(mesh.faces):,}")
    print(f"  • Bounds:   {mesh.bounds.tolist()}")
    print("[*] Opening interactive 3D viewer window (Use mouse to rotate, pan, and zoom)...")
    mesh.show()

if __name__ == '__main__':
    main()
