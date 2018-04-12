import mshr
import dolfin as dlf

p1 = dlf.Point()
p2 = dlf.Point(1, 0.5)
p3 = dlf.Point(0, 0.5)
p4 = dlf.Point(0.5, 1)

domain = mshr.Rectangle(p1, p2) + mshr.Rectangle(p3, p4)
mesh = mshr.generate_mesh(domain, 25)

dlf.File("lshape-mesh-fine.xml.gz") << mesh
