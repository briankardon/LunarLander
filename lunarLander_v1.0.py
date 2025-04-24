import cv2
import numpy as np
from scipy import ndimage
import sys

KEY_UP = 2490368
KEY_DOWN = 2621440
KEY_LEFT = 2424832
KEY_RIGHT = 2555904

KEY_UP_MAC = 63232
KEY_DOWN_MAC = 63233
KEY_LEFT_MAC = 63234
KEY_RIGHT_MAC = 63235


def copy_to_destination(src, dst, start_idx, calculate_normal=False):
    """
    Copy an N-dimensional numpy array src to a destination array dst at a given start index.

    Thanks ChatGPT!

    Parameters:
    - src (ndarray): The source array to copy from.
    - dst (ndarray): The destination array to copy to.
    - start_idx (tuple): The starting index in the destination array.

    Returns:
    - The amount of overlap with other nonzero pixels already in the dst array
        and the mean normal direction from the overlap to the src array center
    """
    # Ensure that start_idx is a tuple
    if not isinstance(start_idx, (tuple, list)):
        raise ValueError("start_idx must be a tuple of indices.")

    # Determine the slices for both arrays
    slices = []

    in_range = True
    normal = None
    overlap = 0

    for dim in range(len(src.shape)):
        start = start_idx[dim]
        stop = start + src.shape[dim]

        # Adjust start and stop indices for the destination array
        if start < 0:
            start = 0
        if stop > dst.shape[dim]:
            stop = dst.shape[dim]

        if start == stop:
            in_range = False
            return overlap, normal, in_range

        slices.append(slice(start, stop))

    # Create a mask for valid indices
    src_slices = tuple(slice(max(0, -start_idx[dim]), min(src.shape[dim], dst[slices[dim]].shape[dim]))
                       for dim in range(len(src.shape)))

    # Extract the source array
    src_array = src[tuple(src_slices)]
    dst_array = dst[tuple(slices)]

    try:
        # Get amount of overlap
        overlap = np.logical_and(dst_array, src_array).sum(axis=(0, 1)).max()

        # Perform the copy operation
        dst[tuple(slices)] = np.maximum(dst_array, src_array)
    except ValueError:
        overlap = 0

    if calculate_normal and overlap > 0:
        cm = np.array(ndimage.center_of_mass(dst_array))
        center = (np.array(dst_array.shape) - 1) / 2
        normal = center - cm
        normal = normal[0:2]
        normal = normal / np.linalg.norm(normal)
        # print('cm=', cm, 'center', center, 'normal', normal)

    return overlap, normal, in_range

class Sprite:
    def __init__(self, position, orientation, velocity, angularVelocity, size):
        self.position = np.array(position)
        self.orientation = orientation
        self.angularVelocity = angularVelocity
        self.velocity = np.array([velocity])
        self.size = size
        self.image = None
        self.destructing = False
        self.life = 100
        self.damageRate = 0.25
        self.onScreen = False

    def generateImage(self):
        radius = int(self.size / 2)
        self.image = np.zeros([radius, radius, 3], dtype='uint8')

    def getRotatedImage(self, angle=None, image=None):
        if image is None:
            image = self.image
        if angle is None:
            angle = self.orientation
        if angle == 0:
            return image.copy()
        return ndimage.rotate(image, -angle, reshape=True)
    def getScaledImage(self, scale=1, image=None):
        if image is None:
            image = self.image
        if scale == 1:
            return image.copy()
        newSize = [int(s / scale) for s in image.shape[0:2]]
        return cv2.resize(image, dsize=newSize, interpolation=cv2.INTER_CUBIC)

    def draw(self, screen, scale=1, coordinateTransform=None):
        h, w, _ = self.image.shape
        if self.destructing:
            disintegrateImage(self.image, damageCount=int(w*h * self.damageRate))
            self.life -= 1
        if self.life <= 0:
            return 0

        if coordinateTransform is not None:
            x, y = coordinateTransform(*(self.position - [w, h]))
        image = self.getScaledImage(image=self.image, scale=scale)
        image = self.getRotatedImage(image=image, angle=self.orientation)
        overlap, _, self.onScreen = copy_to_destination(
            image,
            screen,
            [int(y), int(x), 0])
        return overlap

    def destroy(self):
        self.destructing = True

class Asteroid(Sprite):
    def __init__(self, initialPosition, meanRadius=10, meanSpeed=2, color=[255, 255, 255]):
        Sprite.__init__(self, position=initialPosition, orientation=0, velocity=None, angularVelocity=0, size=None)
        speed = np.random.poisson(lam=meanSpeed)
        angle = np.random.uniform(0, np.pi)
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        self.velocity = np.array([vx, vy], dtype='float')
        self.x = None
        self.y = None
        self.color = color

        radius = np.random.poisson(lam=meanRadius)
        self.generateCoordinates(radius)
        self.generateImage()

    def generateImage(self):
        super().generateImage()
        radius = int(self.size/2)
        x = (self.x + radius).astype('int').tolist()
        y = (self.y + radius).astype('int').tolist()
        pts = np.array(list(zip(x, y))).reshape((-1, 1, 2))
        cv2.fillPoly(self.image, [pts], color=self.color)

    def generateCoordinates(self, radius):
        numPoints = max([3, int(np.ceil(2*np.pi*radius / 8))])
        angles = np.random.uniform(0, 2*np.pi, size=numPoints)
        angles.sort()
        radii = radius * np.random.uniform(0.8, 1.2, size=numPoints)
        self.x = radii * np.cos(angles)
        self.y = radii * np.sin(angles)
        # Get actual radius
        radius = np.sqrt(2) * max(max(self.x) - min(self.x), max(self.y) - min(self.y)) / 2
        self.size = radius * 2

class PowerUp(Sprite):
    def __init__(self, initialPosition, type, image, meanSpeed=2, maxAngularVelocity=2):
        Sprite.__init__(self, position=initialPosition, orientation=0, velocity=None, angularVelocity=0, size=None)
        self.image = image
        speed = np.random.poisson(lam=meanSpeed)
        angle = np.random.uniform(0, np.pi)
        self.angularVelocity = np.random.uniform(-maxAngularVelocity, maxAngularVelocity)
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        self.velocity = np.array([vx, vy], dtype='float')
        self.damageRate = 0.1
        self.captured = False

class Lander(Sprite):
    pass

POWERUP_FILES = {
    'Airbags':'assets/PowerUp_Airbags.png',
    'ExtraLife':'assets/PowerUp_ExtraLife.png',
    'Fuel':'assets/PowerUp_Fuel.png',
    'Missiles':'assets/PowerUp_Missiles.png'
    }

# Load images
POWERUP_IMAGES = {}
for type in POWERUP_FILES:
    POWERUP_IMAGES[type] = cv2.imread(POWERUP_FILES[type])

class Message:
    def __init__(self,
                text='',
                duration=100,
                color=[255, 255, 255],
                color2=None,
                size=0.5,
                flashFrequency=None,
                font=cv2.FONT_HERSHEY_SIMPLEX,
                thickness=1):
        self.text = text
        self.duration = duration
        self.color = color
        self.color2 = color2
        self.colorPairs = None
        self.size = size
        self.flashFrequency = flashFrequency
        self.font = font
        self.thickness = thickness
        self.textSize = cv2.getTextSize(self.text, self.font, self.size, self.thickness)
        self.makeColorPairs()

    def makeColorPairs(self):
        if self.color2 is not None:
            self.colorPairs = list(zip(self.color, self.color2))
        else:
            self.colorPairs = None

    def draw(self, screen, time=0):
        if self.duration > 0:
            h, w, _ = screen.shape
            x = int((w - self.textSize[0][0])/2)
            y = int((h - self.textSize[0][1])/2)
            if self.flashFrequency is None:
                color = self.color
            else:
                f = (np.sin(time * self.flashFrequency) + 1) / 2
                if self.colorPairs is None:
                    self.makeColorPairs()
                color = [int(c1 * f + c2 * (1-f)) for c1, c2 in self.colorPairs]
            cv2.putText(screen, self.text, (x, y), self.font, self.size, color, self.thickness)
            self.duration -= 1

class LunarLander:
    def __init__(self):
        self.w = 1000
        self.h = 600
        self.screen = None
        self.window_name = 'screen'
        self.screenCenter = np.array([0, 0])
        self.screenScale = 1
        self.groundHeight = 200
        self.maxGroundDepth = 100
        self.groundX = None
        self.groundY = None
        self.padX0s = None
        self.padX1s = None
        self.padY0s = None
        self.padY1s = None
        self.landerPosition = np.array([0, 90], dtype='float')
        self.landerVelocity = np.array([0, 0], dtype='float')
        self.landerSpeed = 0   # Only for readout purposes
        self.landerAngularVelocity = 0
        self.landerOrientation = 0
        self.landerRotationMatrix = None
        self.landerRotationSpeed = 5
        self.landerThrusterPower = 0.1
        self.landerMaxFuel = 15
        self.landerFuel = self.landerMaxFuel
        self.gravity = np.array([0, 0.002], dtype='float')
        self.landerImage = None
        self.landerImageBackup = None
        self.landerRadius = None
        self.landerThumbnail = None
        self.landerThumbnailScale = None
        self.landerThumbnailRadius = None
        self.landerStatus = 'flying'  # One of flying, crashed, landed
        self.exhaustPlume = 0
        self.HUDFont = cv2.FONT_HERSHEY_SIMPLEX
        self.HUDFontSize = 0.5
        self.HUDColor = [255, 200, 200]
        self.padColor = [150, 150, 100]
        self.groundColor = [155, 155, 155]
        self.takeoffGrace = 0
        self.asteroids = []
        self.time = 0
        self.messages = []
        self.lives = 3
        self.lastPadIdx = None
        self.bouncy = False
        self.powerups = []
        self.offscreenKillDistanceSquared = (self.h)**2 + (self.w)**2

        self.createLanderRotationMatrix()
        self.createLanderImage()
        self.generateGround()
        startingPadIdx = self.chooseStartingPad()
        self.moveToPad(startingPadIdx)
        self.setLanded(startingPadIdx, showMessage=False)
        self.showBeginMessage()

    def showBeginMessage(self):
        self.messages.append(
            Message(
                text="Begin!",
                duration=100,
                color=[255, 150, 150],
                color2=[150, 50, 50],
                flashFrequency=0.1,
                size=1)
            )

    def run(self):
        while True:
            if self.isCrashed():
                self.disintegrateLander()

            self.causeEvents()

            self.applyPhysics()

            statusNotes, onScreenPadIdx, visibleRange = self.drawPhysicsStuff()
            if self.takeoffGrace > 0:
                # Don't worry about crashing or landing if we're just taking off
                self.takeoffGrace -= 1
                self.setFlying()
            self.drawDecorations(onScreenPadIdx=onScreenPadIdx)

            cv2.imshow(self.window_name, self.screen)
            key = cv2.waitKeyEx(10)
            if key != -1:
                self.handleKeyPress(key)

            self.updateScreenView(visibleRange)

            self.time += 1

    def causeEvents(self):
        c = np.random.uniform()
        if c < 0.0001:
            self.messages.append(
                Message(
                    text="Asteroid alert!",
                    duration=50,
                    color=[150, 150, 255],
                    color2=[50, 50, 255],
                    flashFrequency=0.1,
                    size=1)
                )
            self.addAsteroids()
        c = np.random.uniform()
        # if c < 0.0001:
        #     self.createPowerUp()

    def createPowerUp(self):
        position = self.landerPosition + [np.random.uniform(-100, 100), -self.h * self.screenScale + np.random.uniform(-self.h/5, self.h/5)]
        type = np.random.choice(list(POWERUP_IMAGES.keys()))
        self.powerups.append(
            PowerUp(position, type, POWERUP_IMAGES[type])
        )

    def addAsteroids(self, N=15):
        for k in range(N):
            position = self.landerPosition + [np.random.uniform(-100, 100), -self.h * self.screenScale + np.random.uniform(-self.h/5, self.h/5)]
            newAsteroid = Asteroid(position, meanRadius=30, meanSpeed=0.5, color=self.groundColor)
            self.asteroids.append(newAsteroid)

    def generateGround(self, nPads=10):
        rng = np.random.default_rng()
        N = 1000
        minDeltaX = 0
        maxDeltaX = 40
        meanDeltaX = 20
        deltaY = rng.integers(low=-40, high=40, size=N)
        deltaX = rng.integers(low=minDeltaX, high=maxDeltaX, size=N)

        self.groundX = np.zeros(N)
        self.groundY = np.zeros(N)

        x = -1000
        y = self.groundHeight
        for k in range(N):
            self.groundX[k] = x
            self.groundY[k] = y
            x += deltaX[k]
            y += deltaY[k]
            if y > self.groundHeight + self.maxGroundDepth or y < 0:
                y -= 2*deltaY[k]

        # Generate chasms
        nChasms = 10
        maxW = 25
        for k in range(nChasms):
            centerIdx = np.random.randint(maxW, N-maxW)
            idx0 = max([np.random.randint(centerIdx - int(maxW/2), centerIdx - int(maxW/4)), 0])
            idx1 = min([np.random.randint(idx0 + int(maxW/2), idx0 + maxW), N])
            c = self.groundX[[idx0, idx1]].mean()
            for idx in range(idx0, idx1):
                x = self.groundX[idx]
                r = abs(x - c)
                self.groundY[idx] += 5000 / (r + 1)**0.5

        # Generate mountains
        nMountains = 0
        minW = 25
        maxW = 80
        minHeight=100
        maxHeight=700
        for k in range(nMountains):
            centerIdx = np.random.randint(maxW, N-maxW)
            w = np.random.randint(minW, maxW)
            h = np.random.randint(minHeight, maxHeight)
            idx0 = max([centerIdx - int(w/2), 0])
            idx1 = min([centerIdx + int(w/2), N])
            c = self.groundX[[idx0, idx1]].mean()
            for idx in range(idx0, idx1):
                x = self.groundX[idx]
                r = abs(x - c)
                self.groundY[idx] -= h * 2**(-(r/1000)**2)


        # Generate caves


        # Generate pads
        minX = self.groundX.min()
        maxX = self.groundX.max()
        deltaX = maxX - minX
        padXs = np.arange(minX, maxX, deltaX / (nPads+2))
        padXs = np.delete(padXs, [0, len(padXs)-1])

        meanDist = np.mean(np.diff(padXs))
        variation = int(0.2 * meanDist)
        padXs = padXs + rng.integers(low=-variation, high=variation, size=nPads)
        padIndices = np.argmax(np.diff(np.subtract.outer(self.groundX, padXs) > 0, axis=0), axis=0)

        self.padX0s = np.zeros(nPads)
        self.padY0s = np.zeros(nPads)
        self.padX1s = np.zeros(nPads)
        self.padY1s = np.zeros(nPads)
        padSize = int(self.landerRadius*5 / meanDeltaX)
        for k, padIdx in enumerate(padIndices):
            self.padX0s[k] = self.groundX[padIdx-padSize:padIdx+padSize].min()
            self.padX1s[k] = self.groundX[padIdx-padSize:padIdx+padSize].max()
            self.padY1s[k] = self.groundY[padIdx-padSize:padIdx+padSize].min()
            self.padY0s[k] = self.groundY[padIdx-padSize:padIdx+padSize].max() + 1

    def chooseStartingPad(self):
        padIdx = int(np.round(len(self.padX0s)/2))
        return padIdx

    def moveToPad(self, padIdx):
        padCenterX = (self.padX0s[padIdx] + self.padX1s[padIdx])/2
        padTopY = self.padY1s[padIdx] - self.landerRadius * 0.8
        self.landerPosition = np.array([padCenterX, padTopY], dtype='float')
        self.setLanderOrientation(0)
        self.landerAngularVelocity = 0
        self.screenCenter[0] = padCenterX

    def createLanderImage(self):
        self.landerImage = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 1, 0, 0],
                [0, 1, 0, 1, 1, 0, 1, 0],
                [1, 0, 0, 1, 1, 0, 0, 1],
                [1, 0, 1, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
            ], dtype='uint8') * 255
        # Pad array so lander can be rotated without getting cropped
        maxSize = max(self.landerImage.shape)
        rotationPadding = int(np.ceil(maxSize * (np.sqrt(2)-1)/2))
        self.landerImage = np.pad(self.landerImage, pad_width=((rotationPadding, rotationPadding), (rotationPadding, rotationPadding)))

        # Add 3 color channels to lander image
        self.landerImage = np.stack((self.landerImage, self.landerImage, self.landerImage), axis=2)

        # Scale up lander image so it's bigger
        scale = 3
        newSize = [s*scale for s in self.landerImage.shape[0:2]]
        self.landerImage = cv2.resize(self.landerImage, dsize=newSize, interpolation=cv2.INTER_CUBIC)
        self.landerRadius = max(newSize)/2

        self.landerThumbnailScale = self.landerRadius*2/15
        self.landerThumbnail = self.getScaledLander(self.landerImage, 1/self.landerThumbnailScale)
        self.landerThumbnailRadius = self.landerRadius / self.landerThumbnailScale

        self.landerImageBackup = self.landerImage.copy()

    def disintegrateLander(self):
        disintegrateImage(self.landerImage)

    def checkStatus(self, overlap, normal):
        touching = overlap > 5
        notes = ''
        if self.takeoffGrace > 0 or self.isCrashed():
            return notes
        if touching:
            # Hit something
            condition1 = self.landerPosition[0] - self.landerRadius*0.8 >= self.padX0s
            condition2 = self.landerPosition[0] + self.landerRadius*0.8 <= self.padX1s
            condition3 = np.abs(self.landerPosition[1] + self.landerRadius - self.padY1s) < 5
            if np.any(np.logical_and.reduce([condition1, condition2, condition3])):
                # On target, how's speed?
                speedCondition = self.landerSpeed < 0.35
                if speedCondition:
                    # Speed is ok, how's angle
                    angleCondition = abs(self.landerOrientation) < 5 or abs(self.landerOrientation - 360) < 5
                    if angleCondition:
                        notes = 'Success!'
                        padIdx = np.where(np.logical_and.reduce([condition1, condition2, condition3]))[0].item()
                        self.setLanded(padIdx)
                    else:
                        notes = 'Not straight'
                        if self.bouncy:
                            self.bounce(normal)
                        else:
                            notes = 'Not on target'
                            self.setCrashed()
                else:
                    notes = 'Too fast'
                    if self.bouncy:
                        self.bounce(normal)
                    else:
                        notes = 'Not on target'
                        self.setCrashed()
            else:
                if self.bouncy:
                    self.bounce(normal)
                else:
                    notes = 'Not on target'
                    self.setCrashed()
        else:
            # Still flying
            self.setFlying()
        return notes

    def bounce(self, normal):
        vNormal = self.landerVelocity.dot(normal) * normal
        self.landerVelocity = self.landerVelocity - 2*vNormal

    def drawPhysicsStuff(self):
        self.clearScreen()
        onScreenPadIdx, visibleRange = self.drawGround()
        self.drawAsteroids()
        self.drawPowerUps()
        overlap, normal = self.drawLander()
        if not self.isLanded():
            statusNotes = self.checkStatus(overlap, normal)
        else:
            statusNotes = ''
        return statusNotes, onScreenPadIdx, visibleRange

    def drawPowerUps(self):
        deadPowerups = []
        for k, powerup in enumerate(self.powerups):
            overlap = powerup.draw(self.screen, scale=self.screenScale, coordinateTransform=self.world2Screen)
            if overlap > 5:
                powerup.destroy()
            if powerup.life <= 0:
                deadPowerups.append(k)

        # Remove dead powerups from list
        if len(deadPowerups) > 0:
            self.powerups = [powerup for k, powerup in enumerate(self.powerups) if k not in deadPowerups]

    def drawAsteroids(self):
        deadAsteroids = []
        for k, asteroid in enumerate(self.asteroids):
            overlap = asteroid.draw(self.screen, self.world2Screen)
            if overlap > 5:
                asteroid.destroy()
            if asteroid.life <= 0:
                deadAsteroids.append(k)

        # Remove dead asteroids from list
        if len(deadAsteroids) > 0:
            self.asteroids = [asteroid for k, asteroids in enumerate(self.asteroids) if k not in deadAsteroids]

    def drawDecorations(self, onScreenPadIdx=[]):
        self.drawExhaustPlume()
        self.drawHUD()
        self.drawPadDecorations(onScreenPadIdx)
        self.drawMessage()

    def drawMessage(self):
        if len(self.messages) > 0:
            self.messages[0].draw(self.screen, time=self.time)
            if self.messages[0].duration <= 0:
                self.messages.pop(0)

    def drawPadDecorations(self, onScreenPadIdx):
        # Draw flashing pad lights
        if len(onScreenPadIdx) > 0:
            lightRadius = int(round(3 / self.screenScale))
            padX0s, padY0s = self.world2Screen(self.padX0s[onScreenPadIdx], self.padY0s[onScreenPadIdx], integer=True)
            padX1s, padY1s = self.world2Screen(self.padX1s[onScreenPadIdx], self.padY1s[onScreenPadIdx], integer=True)
            leftCenterX = padX0s + lightRadius
            rightCenterX = padX1s - lightRadius
            centerX = ((padX0s + padX1s)/2).astype('int')
            centerY = padY1s - 1
            lightIntensity = int(100 + 155*(np.sin(self.time/10)+1)/2)
            if self.isLanded():
                lightColor = [0, lightIntensity, 0]
            else:
                lightColor = [0, 0, lightIntensity]
            for idx in onScreenPadIdx:
                cv2.ellipse(self.screen,
                    center=np.array([leftCenterX[idx], centerY[idx]], dtype='int'),
                    axes=[lightRadius, lightRadius],
                    angle=0,
                    startAngle=180,
                    endAngle=360,
                    color=lightColor,
                    thickness=-1)
                cv2.ellipse(self.screen,
                    center=np.array([rightCenterX[idx], centerY[idx]], dtype='int'),
                    axes=[lightRadius, lightRadius],
                    angle=0,
                    startAngle=180,
                    endAngle=360,
                    color=lightColor,
                    thickness=-1)
                cv2.putText(self.screen, str(idx), (centerX[idx], centerY[idx] + 15), self.HUDFont, self.HUDFontSize / self.screenScale, [0, 50, 50])

    def drawHUD(self):
        wHUD = int(0.2 * self.w)
        hHUD = int(0.2 * self.h)
        # HUD outline
        cv2.rectangle(self.screen, (0, 0), (wHUD, hHUD), self.HUDColor)

        # Fuel gauge label
        cv2.putText(self.screen, 'Fuel', (10, 15), self.HUDFont, self.HUDFontSize, self.HUDColor)
        xFuel = int(wHUD*0.05)
        yFuel = 20
        wFuel = int(wHUD*0.75)
        hFuel = 10
        fFuel = self.landerFuel / self.landerMaxFuel
        wFuelRemaining = int(wFuel * fFuel)

        # Fuel level
        if fFuel < 0.15:
            fuelColor = [0, 0, 255]
        elif fFuel < 0.3:
            fuelColor = [0, 255, 255]
        else:
            fuelColor = [0, 255, 0]
        cv2.rectangle(self.screen, (xFuel, yFuel), (xFuel + wFuelRemaining, yFuel + hFuel), fuelColor, thickness=-1)
        # Fuel outline
        cv2.rectangle(self.screen, (xFuel, yFuel), (xFuel + wFuel, yFuel + hFuel), self.HUDColor)

        # Velocity readout
        cv2.putText(
            self.screen,
            'Speed:  {s:.02f} m/s'.format(s=self.landerSpeed),
            (10, 45),
            self.HUDFont,
            self.HUDFontSize,
            self.HUDColor
        )

        # Status readout
        cv2.putText(
            self.screen,
            'Status: {status}'.format(status=self.landerStatus),
            (10, 75),
            self.HUDFont,
            self.HUDFontSize,
            self.HUDColor
        )

        # Draw extra lives
        self.drawLander(lander=self.landerThumbnail, position=np.array((10, 90 - int(self.landerThumbnailRadius))), orientation=0, scale=1, worldUnits=False)
        cv2.putText(
            self.screen,
            'x {lives}'.format(lives=self.lives),
            (15 + int(2*self.landerThumbnailRadius), 95),
            self.HUDFont,
            self.HUDFontSize,
            self.HUDColor
        )


    def drawExhaustPlume(self):
        size = min([self.exhaustPlume, 10])

        if size > 0:
            offset = -self.landerRadius*0.6
            pt0 = (self.landerPosition - self.landerRotationMatrix.dot([-size*0.5, offset]))
            pt1 = (self.landerPosition - self.landerRotationMatrix.dot([0, -(self.landerRadius*0.5 + size) + offset]))
            pt2 = (self.landerPosition - self.landerRotationMatrix.dot([size*0.5, offset]))

            x, y = zip(*[pt0, pt1, pt2])

            x, y = self.world2Screen(x, y, integer=True, asList=False)

            pts = np.array(list(zip(x, y))).reshape((-1, 1, 2))

            cv2.polylines(self.screen, [pts], color=[100, 100, 255], isClosed=False)

    def createLanderRotationMatrix(self):
        self.landerRotationMatrix = self.makeRotationMatrix(self.landerOrientation)

    def fireThrusters(self):
        if self.landerFuel > 0:
            self.landerVelocity += self.landerThrusterPower * self.landerRotationMatrix.dot(np.array([0, -1]))
            self.exhaustPlume += 2
            self.landerFuel = max([0, self.landerFuel - self.landerThrusterPower])

    def fireRCS(self, amount):
        if self.landerFuel > 0:
            # Command fire
            self.landerAngularVelocity += amount
            # Auto-stabilize
            self.landerFuel = max([0, self.landerFuel - self.landerThrusterPower/10])

    def useExtraLife(self):
        if self.lives > 0:
            self.lives -= 1
            self.landerFuel = self.landerMaxFuel
            self.resetLanderImage()
            self.moveToPad(self.lastPadIdx)
            self.setLanded(self.lastPadIdx, showMessage=False)
            self.showBeginMessage()

    def resetLanderImage(self):
        self.landerImage = self.landerImageBackup.copy()

    def isLanded(self):
        return self.landerStatus == 'landed'
    def isFlying(self):
        return self.landerStatus == 'flying'
    def isCrashed(self):
        return self.landerStatus == 'crashed'

    def setLanded(self, padIdx, showMessage=True):
        self.landerStatus = 'landed'
        self.lastPadIdx = padIdx
        if showMessage:
            self.messages.append(
                Message(
                    text="Landed!",
                    duration=100,
                    color=[150, 255, 150],
                    color2=[50, 150, 50],
                    flashFrequency=0.1,
                    size=1)
                )
    def setFlying(self):
        self.landerStatus = 'flying'
    def setCrashed(self, showMessage=True):
        self.landerStatus = 'crashed'
        if showMessage:
            self.messages.append(
                Message(
                    text="Crashed!",
                    duration=50,
                    color=[150, 150, 255],
                    color2=[50, 50, 150],
                    flashFrequency=0.1,
                    size=1)
                )
            if self.lives > 0:
                self.messages.append(
                    Message(
                        text="Press enter to try again!",
                        duration=150,
                        color=[150, 150, 255],
                        color2=[50, 50, 150],
                        flashFrequency=0.1,
                        size=1)
                    )
            else:
                self.messages.append(
                    Message(
                        text="Game over!",
                        duration=150,
                        color=[150, 150, 255],
                        color2=[50, 50, 150],
                        flashFrequency=0.1,
                        size=1)
                    )

    def handleKeyPress(self, key):
        # if key != -1:
        #     print(key)

        if self.isCrashed():
            if key == 13: # Enter
                # Use extra life and reset
                if self.lives > 0:
                    self.useExtraLife()
                else:
                    sys.exit()

        if key == 120:   # x
            pass

        if key == 113:   # q
            sys.exit()

        if not self.isCrashed():
            if key in [KEY_UP, KEY_UP_MAC]:  # Up
                self.fireThrusters()
                if self.isLanded():
                    self.takeoffGrace = 10
                    # Extra kick
                    self.setFlying()
                    self.fireThrusters()
            if key == 97: # a
                # self.addAsteroids()
                self.createPowerUp()

            if self.isLanded():
                # Don't respond to other keys if landed
                return

            if key in [KEY_LEFT, KEY_LEFT_MAC]:  # Left
                self.fireRCS(-1)
            if key in [KEY_DOWN, KEY_DOWN_MAC]:  # Down
                pass
            if key in [KEY_RIGHT, KEY_RIGHT_MAC]:  # Right
                self.fireRCS(1)

    def makeRotationMatrix(self, angle):
        c = np.cos(angle * np.pi / 180)
        s = np.sin(angle * np.pi / 180)
        return np.array([[c, -s], [s, c]])

    def setLanderOrientation(self, angle):
        self.landerOrientation = angle
        self.createLanderRotationMatrix()

    def applyPhysics(self):
        if self.isLanded():
            self.landerVelocity = np.array([0, 0], dtype='float')
            self.landerSpeed = 0
            self.setLanderOrientation(0)
            self.exhaustPlume = 0
            # Refuel
            self.landerFuel = min([self.landerMaxFuel, self.landerFuel + 0.02])
            return

        if self.isFlying():
            if not (self.landerAngularVelocity == 0):
                self.setLanderOrientation(np.mod(self.landerOrientation + self.landerAngularVelocity, 360))
                self.landerAngularVelocity = self.landerAngularVelocity * 0.9
                if abs(self.landerAngularVelocity) < 0.1:
                    self.landerAngularVelocity = 0
            self.landerVelocity += self.gravity
            self.landerPosition += self.landerVelocity
            self.exhaustPlume = max([self.exhaustPlume - 0.5, 0])
            self.landerSpeed = np.sqrt(self.landerVelocity.dot(self.landerVelocity))
        else:
            self.landerVelocity = np.array([0, 0], dtype='float')
            self.lanmderSpeewd = 0
            self.exhaustPlume = 0

        for asteroid in self.asteroids:
            asteroid.orientation += asteroid.angularVelocity
            asteroid.velocity += self.gravity
            asteroid.position += asteroid.velocity

        for powerup in self.powerups:
            powerup.orientation += powerup.angularVelocity
            powerup.velocity += self.gravity
            powerup.position += powerup.velocity

    def clearScreen(self):
        self.screen = np.zeros([self.h, self.w, 3], 'uint8')

    def updateScreenView(self, visibleRange):
        x, y = self.world2Screen(*self.landerPosition)
        xMarginFactor = 0.25
        xMargin = self.w * xMarginFactor

        distFromLeftMargin = x - xMargin
        if distFromLeftMargin < 0:
            self.screenCenter[0] += distFromLeftMargin

        distFromRightMargin = x - (self.w - xMargin)
        if distFromRightMargin > 0:
            self.screenCenter[0] += distFromRightMargin

        yMarginFactor = 0.15
        yMargin = self.h * yMarginFactor
        # Distance from lander to top of screen margin
        distFromTopMargin = y - yMargin
        distFromBotMargin = self.h - yMargin - y
        # Distance of highest ground point to bottom of screen margin
        lowestVisiblePoint, highestVisiblePoint = visibleRange
        groundBotMargin = self.h - yMargin - highestVisiblePoint
        # visibleGroundGoodMargin = self.h - 3*yMargin - highestVisiblePoint
        # lowestgroundBotMargin = self.h - yMargin - lowestVisiblePoint

        # print('groundBotMargin', int(groundBotMargin), 'distFromTopMargin', int(distFromTopMargin))

        zoomSpeed = 1.002
        minScreenScale = 1

        output = False
        if distFromTopMargin < 0: # and groundBotMargin > 0:
            # Shift view up
            # print('shifting up')
            self.screenCenter[1] += distFromTopMargin
        elif distFromBotMargin < 0: #and (visibleGroundGoodMargin < 0 or distFromBotMargin < 0):
            # Shift view down
            # print('shifting down')
            self.screenCenter[1] -= distFromBotMargin

        if groundBotMargin < 0:
            # Zoom out
            # print('zooming out')
            self.screenScale *= zoomSpeed
        elif self.screenScale > minScreenScale: #distFromTopMargin >= 0 and groundBotMargin >= 0 and self.screenScale > minScreenScale:
            # zoom in
            # print('zooming in')
            self.screenScale = self.screenScale / zoomSpeed
            self.screenCenter[1] += 1

        # Limit zoom-in to 1
        self.screenScale = max([minScreenScale, self.screenScale])
        # print(self.screenScale)

        # if output:
        #     print('highestVisiblePoint ', highestVisiblePoint)
        #     print('y                   ', y)
        #     print('self.h              ', self.h)
        #     print('yMargin             ', yMargin)
        #     print('(self.h - 2*yMargin)', (self.h - 2*yMargin))
        #     print('(self.h - 3*yMargin)', (self.h - 3*yMargin))

    def world2Screen(self, worldX, worldY, integer=False, asList=False):
        screenX = (np.array(worldX) - self.screenCenter[0]) / self.screenScale + self.w / 2
        screenY = (np.array(worldY) - self.screenCenter[1]) / self.screenScale + self.h / 2
        if integer:
            try:
                screenX = screenX.astype('int')
            except:
                screenX = [int(v) for v in x]
            try:
                screenY = screenY.astype('int')
            except:
                screenY = [int(v) for v in y]
        if asList:
            try:
                screenX = screenX.tolist()
            except:
                pass
            try:
                screenY = screenY.tolist()
            except:
                pass
        return screenX, screenY

    def drawGround(self):
        groundX, groundY = self.world2Screen(self.groundX, self.groundY)
        groundX = groundX.astype('int')
        groundY = groundY.astype('int')
        onScreen = np.logical_and.reduce([groundX >= 0, groundX <= self.w, groundY >= 0, groundY <= self.h])
        onScreenIdx = np.nonzero(onScreen)[0]
        if len(onScreenIdx) > 0:
            firstIdx = max([1, onScreenIdx.min()-1])
            lastIdx = min([len(groundX), onScreenIdx.max()+1])
            groundX = np.append(groundX[firstIdx:lastIdx+1], [groundX[lastIdx+1], groundX[firstIdx]])
            groundY = np.append(groundY[firstIdx:lastIdx+1], [self.h*2, self.h*2])
            pts = np.array(list(zip(groundX, groundY))).reshape((-1, 1, 2))

            highestVisiblePoint = groundY.min()
            lowestVisiblePoint = groundY.max()
            visibleRange = [lowestVisiblePoint, highestVisiblePoint]
            # lowestVisiblePoint = max([0, groundY.max()
            cv2.fillPoly(self.screen, [pts], color=self.groundColor)
        else:
            visibleRange = [self.h, self.h]

        # Draw pads
        padX0s, padY0s = self.world2Screen(self.padX0s, self.padY0s, integer=True)
        padX1s, padY1s = self.world2Screen(self.padX1s, self.padY1s, integer=True)
        onScreenIdx = np.nonzero(np.logical_and(np.logical_or(padX0s >= 0, padX1s <= self.w), np.logical_or(padY0s >= 0, padY1s <= self.h)))[0].tolist()
        for idx in onScreenIdx:
            cv2.rectangle(self.screen, (padX0s[idx], padY0s[idx]), (padX1s[idx], padY1s[idx]), color=self.groundColor, thickness=-1)
        return onScreenIdx, visibleRange # Pass out visible pads for convenient drawing of decorations

    def getRotatedLander(self, landerImage, angle):
        if angle == 0:
            return self.landerImage
        return ndimage.rotate(landerImage, -angle, reshape=False)

    def getScaledLander(self, landerImage, scale):
        newSize = [int(s * scale) for s in landerImage.shape[0:2]]
        return cv2.resize(landerImage, dsize=newSize, interpolation=cv2.INTER_CUBIC)

    def drawLander(self, position=None, orientation=None, scale=None, worldUnits=True, lander=None):
        if position is None:
            position = self.landerPosition
        if orientation is None:
            orientation = self.landerOrientation
        if scale is None:
            scale = self.screenScale

        radius = self.landerRadius / scale
        if worldUnits:
            x, y = self.world2Screen(*(position - [radius, radius]))
        else:
            x, y = position
        if lander is None:
            lander = self.landerImage
            lander = self.getRotatedLander(lander, orientation)
            lander = self.getScaledLander(lander, 1/scale)
        overlap, normal, _ = copy_to_destination(lander, self.screen, (int(y), int(x), 0), calculate_normal=self.bouncy)
        return overlap, normal

def disintegrateImage(image, damageCount=50):
    rng = np.random.default_rng()
    radius_x = int(image.shape[0]/2)
    radius_y = int(image.shape[1]/2)
    x_dev = rng.integers(low=-3, high=3, size=damageCount)
    y_dev = rng.integers(low=-3, high=3, size=damageCount)
    x_start = rng.integers(low=-(radius_x-1), high=(radius_x-1), size=damageCount)
    y_start = rng.integers(low=-(radius_y-1), high=(radius_y-1), size=damageCount)
    x_end = x_start*2 + x_dev
    y_end = y_start*2 + y_dev

    x_start = (x_start + radius_x).astype('int')
    y_start = (y_start + radius_y).astype('int')
    x_end = (x_end + radius_x).astype('int')
    y_end = (y_end + radius_y).astype('int')

    out_of_range = np.logical_or.reduce([x_end < 0, x_end >= image.shape[0], y_end < 0, y_end >= image.shape[1]])

    vals = image[x_start, y_start, :]

    vals = np.delete(vals, out_of_range, axis=0)
    x_end = np.delete(x_end, out_of_range)
    y_end = np.delete(y_end, out_of_range)

    # Copy pixels
    image[x_end, y_end, :] = vals
    # Clear pixels
    image[x_start, y_start, :] = [0, 0, 0]

ll = LunarLander()

while True:
    ll.run()
