import numpy as np
import quaternion

if __name__ == '__main__':
    q = np.quaternion(0.1, 0.2, 0.3, 0.4)
    r = hex(int(q.real*255))
    x = hex(int(q.x*255))
    y = hex(int(q.y*255))
    z = hex(int(q.z*255))
    return '[{} {} {} {}]'.format(r, x, y, z)