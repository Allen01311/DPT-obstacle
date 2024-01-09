#!/usr/bin/env python
import math
class space_mod:

  def __init__(self):
    
    #self.missionPoint = [[24.983332, 121.572034],[24.98272505, 121.572466],[24.982302,121.572039],[24.982022, 121.571627]] # three stage
    #self.missionPoint = [[24.983332, 121.572034],[24.98272505, 121.572466],[24.982022, 121.571627],[24.981858, 121.571947333]] # delivery
    #self.missionPoint = [[24.9848669152, 121.572756628],[24.9848375782, 121.572870165],[24.9845588075, 121.572601239],[24.9844943067, 121.572629345],
    #                     [24.9848125371, 121.572936339],[24.984771591, 121.572987168],[24.9844433247, 121.572670492],[24.9844199853, 121.572738306],
    #                     [24.9846967166, 121.573005266],[24.9846334912, 121.573034602],[24.9843966467, 121.57280612],[24.9844126215, 121.57291186],[24.9845718653, 121.573065481],
    #                     [24.9848669152, 121.572756628]] # Region scan 5m
    #self.missionPoint = [[24.9848669152, 121.572756628],[24.9848177216, 121.572816008],[24.9845903475, 121.572596663],[24.9845064967, 121.5726332],
    #                     [24.9848081827, 121.572924233],[24.984749134, 121.572984699],[24.9844449113, 121.572691217],[24.9844145697,121.572779375],
    #                     [24.9846569879, 121.573013233],[24.9845768742, 121.573053377],[24.9844109583, 121.572893319],[24.9848669152, 121.572756628]] # Region scan 6.5m
    #self.missionPoint = [[24.985577, 121.574881],[24.9852737399, 121.575053195],[24.9862121099, 121.574448289],[24.9862495872, 121.574321845],
    #                     [24.985109217, 121.575056969],[24.9850422387, 121.574997864],[24.9862240427, 121.574236029],[24.9861984981, 121.574150212],
    #                     [24.9849763506, 121.574938054],[24.9849104617, 121.574878245],[24.9861556052, 121.574075579],[24.9861126693, 121.574000973],
    #                     [24.9848445736, 121.574818436],[24.9848392655, 121.574719576],[24.9860697333, 121.573926369],[24.9859800336, 121.57388191],
    #                     [24.9848423381, 121.574615314],[24.9848454097, 121.57451105],[24.9858715561, 121.573849556],[24.9857165726, 121.573847181],
    #                     [24.9848528704, 121.574403959],[24.9850144779, 121.574197498],[24.985123951, 121.574126928],[24.985577, 121.574881]] # Region scan court1
    #self.missionPoint = [[24.9857165726, 121.573847181],[24.985123951, 121.574126928],[24.9850144779, 121.574197498],[24.9848528704, 121.574403959],
    #                     [24.9858715561, 121.573849556],[24.9859800336, 121.57388191],[24.9848454097, 121.57451105],[24.9848423381, 121.574615314],
    #                     [24.9860697333, 121.573926369],[24.9861126693, 121.574000973],[24.9848392655, 121.574719576],[24.9848445736, 121.574818436],
    #                     [24.9861556052, 121.574075579],[24.9861984981, 121.574150212],[24.9849104617, 121.574878245],[24.9849763506, 121.574938054],
    #                     [24.9862240427, 121.574236029],[24.9862495872, 121.574321845],[24.9850422387, 121.574997864],[24.985109217, 121.575056969],
    #                     [24.9862121099, 121.574448289],[24.985577, 121.574881],[24.9852737399, 121.575053195],[24.9857165726, 121.573847181]] # Region scan court2
    #self.missionPoint = [[24.9851631791, 121.57504087],[24.9862303492, 121.574352933],[24.9861692705, 121.574156268],[24.9849920185, 121.574915168],
    #                     [24.9848614279, 121.574763315],[24.9860701864, 121.573984103],[24.9858758141, 121.573873365],[24.9848685164, 121.574522709],
    #                     [24.9850747983, 121.574153696],[24.985577, 121.574881]] # Rs court3 15m
    #self.missionPoint = [[24.9859096225, 121.575432641],[24.98605077950113, 121.5754572432088],[24.986205767271034, 121.5754847358512],[24.98636324464785, 121.57551823286144]] #building 24.9863055265,121.575453137(end)
    #self.missionPoint = [[24.986143,121.571810], [24.986469,121.571856],[24.986138, 121.572879]] # pass to baseketball court and return
    #self.missionPoint = [ [24.987194, 121.572529], [24.987889, 121.572335], [24.988349, 121.572037], [24.988966, 121.571780], [24.989496, 121.571780], [24.990224, 121.571898], [24.991558, 121.572057], [24.990224, 121.571898], [24.989496, 121.571780], [24.988966, 121.571780], [24.988349, 121.572037], [24.987889, 121.572335], [24.987194, 121.572529], [24.987272, 121.572913] ] # river return
    self.missionPoint = [[]]
    self.nowIndex = 0
    #self.pos = [0,0,0]
    #self.rpy = [0,0,0]
    #self.velocity = [0,0,0]
    #self.safe_region = []
    #self.path = []
    #self.obstacle_pos = [[9,0.3]]

    #self.home_set = False

  def appendMissionP(self, point):
    self.missionPoint.append(point)

  def getMissionP(self):
    return self.missionPoint[self.nowIndex]

  def getIndex(self):
    return self.nowIndex

  def setIndex(self, i):
    self.nowIndex = i

  def getNextMp(self):
    return self.missionPoint[self.nowIndex+1]

  def getOldMp(self):
    return self.missionPoint[self.nowIndex-1]

  def getMissionLength(self):
    return len(self.missionPoint)

  def agl_btw_points(self, p1, p2):
    """ angle from p1 to p2 (p2 is camera)"""
    p = p1 - p2
    return math.atan(p[0]/p[2]) / 0.0174532925

  def angleConvert(self, angle):
    if angle < 0:
      angle = (angle + 360) % 360
    #angle = 360 - angle
    return angle
    
  def angleConvert2PC(self, angle):
    angle = angle - 359.186654992 + 111.274875601
    if angle < 0:
      angle += 360
    return angle
    
  def angleFromCoordinate(self,lat1, long1, lat2, long2):
    dLon = (long2 - long1)
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    brng = math.atan2(y, x)
    brng = math.degrees(brng)
    brng = (brng + 360) % 360
    brng = 360 - brng
    return brng

  def angleFromPoint(self, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    brng = math.degrees(math.atan2(dy, dx))
    brng = (brng + 360) % 360
    brng = brng - 180
    if brng < 0:
      brng = 360 + brng
    return brng

  def angleFromCoordinate_correct(self, lat1, long1, lat2, long2):
    dLon = (long2 - long1)
    y = math.sin(math.radians(dLon)) * math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(dLon))
    brng = math.atan2(y, x)
    brng = math.degrees(brng)
    if brng < 0:
      brng += 360
    #return brng
    #brng = 360 - brng
    return brng

  def angleFromPoint_correct(self, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    brng = math.degrees(math.atan2(dx, dy))
    brng = (brng + 360) % 360
    #brng = brng - 180
    if brng < 0:
      brng = 360 + brng
    brng = brng - 90
    if brng < 0:
      brng = 360 + brng
    return brng

  def calculateDistance(self, x1,y1, x2,y2):  
     dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
     return dist 

  def p2line_eq(self, P, Q):
    a = Q[1] - P[1]
    b = P[0] - Q[0]
    c = a * (P[0]) + b*(P[1])
    
    return a, b, c

  def slove_x(self, a ,b, c, x, y):
    if a != 0:
      return (-(b*y)+c) / a
    elif a == 0:
      print("horizontal line need slove_y!!!")

  def slove_y(self, a, b, c, x, y):
    if b != 0:
      return (-(a*x)+c) / b
    elif b == 0:
      print("vertical line need slove_x!!!")

  def convertRadToAngle(self,rad):
    # 0 ~ 180
    if rad < 0:
      return -rad / (math.pi/180.0)
    # 181 ~ 360
    else:
      return -(rad - math.pi) / (math.pi/180.0) + 180.0

  def convertAngleToRad(self,angle):
    if angle > 180:
      return math.pi - (angle - 180) * (math.pi/180.0)
    else:
      return -angle * (math.pi/180.0)
