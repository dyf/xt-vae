import vtk
import numpy as np

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def build_geo(mc, file_name, sample_size, target_dim=3):
        embeds, colors = mc.embed(target_dim=target_dim)
        
        rng = np.random.default_rng()
        idx = rng.choice(embeds.shape[0], sample_size, replace=False)

        embeds = embeds[idx]
        colors = colors[idx]

        append = vtk.vtkAppendPolyData()
        writer = vtk.vtkPLYWriter()

        for i in range(embeds.shape[0]):
            if i % 5000 == 0:
                print(f"geo {i}")
            p = embeds[i]
            c = hex_to_rgb(colors[i])            

            s = vtk.vtkSphereSource()
            s.SetRadius(0.02)
            s.SetCenter(p[0], p[1], p[2])
            s.SetThetaResolution(16)
            s.SetPhiResolution(8)
            s.Update()

            pd = s.GetOutput()
            ca = vtk.vtkUnsignedCharArray()
            ca.SetNumberOfComponents(3)
            ca.SetName("colors")

            points = pd.GetPoints()
            for i in range(points.GetNumberOfPoints()):
                ca.InsertNextTuple3(c[0], c[1], c[2])

            pd.GetPointData().AddArray(ca)
            
            append.AddInputData(pd)

        append.Update()

        writer.SetFileName(file_name)
        writer.SetArrayName("colors")
        writer.SetFileTypeToBinary()

        writer.SetInputConnection(append.GetOutputPort())
        writer.Write()