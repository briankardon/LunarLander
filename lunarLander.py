import pygame
import numpy as np
from scipy import ndimage
import sys
from pathlib import Path
import traceback
from copy import copy
from collections import deque
from math import pi as PI

pygame.init()

def cropImage(surface):
    originalSize = surface.size
    image = np.zeros(originalSize + (3,), dtype='uint8')
    pygame.pixelcopy.surface_to_array(image, surface)
    x, y = np.meshgrid(range(originalSize[0]), range(originalSize[1]), indexing='ij')
    mask = image > 0
    # if mask.ndim == 3:
    mask = np.any(mask, axis=2)
    xs = x[mask]
    ys = y[mask]
    x0 = xs.min()
    x1 = xs.max()+1
    y0 = ys.min()
    y1 = ys.max()+1
    newW = x1 - x0
    newH = y1 - y0
    newSurface = pygame.Surface((newW, newH))
    print(image[x0:x1, y0:y1, ...].shape)
    print(newSurface.size)
    pygame.pixelcopy.array_to_surface(newSurface, image[x0:x1, y0:y1, ...])
    return newSurface

def makeRotationMatrix(angle):
    # angle in degrees
    c = np.cos(angle * np.pi / 180)
    s = np.sin(angle * np.pi / 180)
    return np.array([[c, -s], [s, c]])

class Particles:
    def __init__(self, position=np.array([0, 0]), life=100, radius=10, rate=10, colors=np.ones([1, 3])*255, persistence=10):
        self.position = position
        self.life = life
        self.persistence = persistence
        self.radius = radius
        self.rate = rate
        self.colors = colors # Nx3 array of uint8 color values to randomly choose from
        self.maxParticles = int(self.persistence * self.rate)
        self.x = deque(maxlen=self.maxParticles)
        self.y = deque(maxlen=self.maxParticles)
        self.vx = deque(maxlen=self.maxParticles)
        self.vy = deque(maxlen=self.maxParticles)

    def draw(self, screen, coordinateTransform=None):
        if self.isAlive():
            self.life -= 1
            if self.life > 0:
                # Only create new particles if life > 0
                r = np.random.normal(loc=0, scale=self.radius / 2, size=self.rate)
                a = np.random.uniform(0, 2*np.pi, size=self.rate)
                newX = r * np.cos(a) + self.position[0]
                newY = r * np.sin(a) + self.position[1]
                newVx = np.random.uniform(-0.1, 0.1, size=self.rate)
                newVy = np.random.uniform(-0.1, 0.1, size=self.rate)
                outOfRange = [k for k in range(len(r)) if r[k] >= self.radius]
                newX = np.delete(newX, outOfRange)
                newY = np.delete(newY, outOfRange)
                newVx = np.delete(newVx, outOfRange)
                newVy = np.delete(newVy, outOfRange)
                self.x.extend(newX)
                self.y.extend(newY)
                self.vx.extend(newVx)
                self.vy.extend(newVy)
            else:
                # Dying, just get rid of old particles
                self.x = list(self.x)[:-self.rate]
                self.y = list(self.y)[:-self.rate]
                self.vx = list(self.vx)[:-self.rate]
                self.vy = list(self.vy)[:-self.rate]
            if coordinateTransform is not None:
                x, y, valid = coordinateTransform(self.x, self.y, integer=True, asList=False, checkValid=True)
            colorChoices = np.random.randint(0, self.colors.shape[0], size=len(x))
            screen[y[valid], x[valid], :] = self.colors[colorChoices[valid], :]

            for k in range(len(self.x)):
                self.x[k] += self.vx[k]
                self.y[k] += self.vy[k]

    def isAlive(self):
        return not (self.life <= 0 and len(self.x) == 0)

class Sprite(pygame.sprite.Sprite):
    def __init__(self,
                    game,
                    position=np.array([0, 0], dtype='int'),
                    size=None,
                    orientation=None,
                    velocity=None,
                    speed=None,     # If velocity is not given, use this to generate a random velocity. If a 2-tuple, will be used as a range, if a single number, will be used as a maximum speed
                    image=None,
                    font=None,  # In case the sprite needs to render text
                    angularVelocity=0,
                    noCollideIDs=[],
                    life=100):
        pygame.sprite.Sprite.__init__(self)
        self.game = game
        if type(font) == str:
            # Font is a string name of a font - try to find it
            font_path = pygame.font.match_font(font)
            self.font = pygame.font.Font(font_path)
        else:
            # Font is a font object, just use it
            self.font = font
        self.particleRegistrar = game.registerParticles
        self.messageRegistrar = game.registerMessage
        self.position = np.array(position)

        self._orientation = self.handleArgumentRange(orientation, 0, 0, 2*np.pi)
        self._lastOrientation = self._orientation
        self._scale = 1
        self._lastScale = self._scale

        self.angularVelocity = self.handleArgumentRange(angularVelocity, 0, -0.1, 0.1)

        if velocity is None:
            # No velocity given, check if speed was given
            if speed is None:
                # Nope, set velocity to zero
                self.velocity = np.array([0.0, 0.0], dtype='float')
            else:
                # Yep, use speed to generate a random velocity
                self.velocity = generateRandomVelocity(speed)
        else:
            self.velocity = np.array(velocity)
        self.thrust = None
        self.size = size
        self.image = image
        self.mask = None
        self.rect = None
        self.collisionMask = None
        self.imageBackup = None
        self.destructing = False
        self.bouncy = False
        self.maxLife = life
        self.life = life
        self.damageRate = 0.15
        self.onScreen = False
        self.noCollideIDs = noCollideIDs
        self.overlapInfoRequired = {
            'calculate_normal':False,
            'calculate_overlap':True,
            'calculate_overlapIDs':True
        }

    def handleArgumentRange(self, arg, defaultVal, minVal, maxVal, dist=np.random.uniform):
        if arg is None:
            return defaultVal
        elif type(arg) in [tuple, list]:
            return dist(*arg)
        else:
            return arg

    def update(self, frozen=False):
        if self.life <= 0:
            return
        if not frozen:
            self.incrementOrientation(self.angularVelocity)
            self.velocity += self.game.gravity
            if self.thrust is not None:
                self.velocity += self.thrust
            self.position += self.velocity
            self.updateRect()
        if self.destructing:
            w = self.image.width
            h = self.image.height
            if w == self.imageBackup.width:
                # First time destructing - expand image to allow debris spread
                newImage = pygame.Surface((int(w*2), int(h*2)))
                newImage.set_colorkey('black')
                rect = self.image.get_rect(center=(int(newImage.width/2), int(newImage.height/2)))
                newImage.blit(self.image, rect)
                self.image = newImage
            disintegrateImage(self.image, damageCount=int(w*h * self.damageRate))
            self.life -= 1

    def updateRect(self):
        if self.image is not None:
            position = self.game.world2Screen(*self.position)
            self.rect = self.image.get_rect(center=position)
    def updateMask(self):
        if self.image is not None:
            self.mask = pygame.mask.from_surface(self.image)

    def setOrientation(self, orientation):
        self._orientation = orientation
        orientationChanged = (self._lastOrientation != orientation)
        if orientationChanged:
            # Orientation changed - update image and rect
            self.image = pygame.transform.rotozoom(
                            self.imageBackup,
                            self.getOrientation(),
                            self.getScale())
            self.updateRect()
        self._lastOrientation = np.mod(orientation, 360)
        return orientationChanged
    def incrementOrientation(self, deltaOrientation):
        self.setOrientation(self._orientation + deltaOrientation)
    def getOrientation(self):
        return self._orientation

    def setScale(self, scale):
        self._scale = scale
        if self._lastScale != scale:
            # Scale changed - update image and rect
            self.image = self.backupImage
        self._lastScale = scale
    def incrementScale(self, deltaScale):
        self.setScale(self._scale + deltaScale)
    def multiplyScale(self, factorScale):
        self.setScale(self._scale * factorScale)
    def getScale(self):
        return self._scale

    def handleSpriteCollision(self, sprite, collisionPoint=None):
        pass

    def handleGroundCollision(self, normal=None):
        if self.bouncy:
            self.bounce(normal)
        else:
            print('sprite of type {t} hit ground'.format(t=type(self)))
            self.destroy()

    def bounce(self, normal):
        vNormal = self.velocity.dot(normal) * normal
        self.velocity = self.velocity - 2*vNormal

    def generateImage(self):
        radius = int(max(self.size) / 2)
        self.image = pygame.Surface([radius*2, radius*2])
        self.backupImage()

    def backupImage(self):
        self.imageBackup = self.image.copy()

    def resetImage(self):
        self.image = self.imageBackup.copy()

    def resetState(self):
        self.destructing = False
        self.life = self.maxLife

    def freeze(self):
        self.velocity = np.array([0, 0], dtype='float')
        self.speed = 0
        self.angularVelocity = 0

    def draw(self, screen, scale=1, orientation=None):
        if orientation is None:
            orientation = self.getOrientation()

        screen.blit(self.image, self.rect)

    def checkOverlap(self, collisionMap, noCollideIDs=[], scale=1,
            coordinateTransform=None, position=None, orientation=None,
            worldUnits=True, collisionMask=None,
            calculate_normal=True, calculate_overlap=True,
            calculate_overlapIDs=True):

            # NEEDS WORK
            return False

    def destroy(self):
        self.destructing = True

class Missile(Sprite):
    def __init__(self, *args, thrust=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.angularVelocity = 0
        fireAngle = (self.getOrientation()-90) * np.pi / 180
        self.thrust = thrust * np.array([np.cos(fireAngle), np.sin(fireAngle)])

        color1 = [0, 0, 200]
        color2 = [150, 150, 255]
        self.explodeColors = createColorRange(color1, color2, 15)

        self.updateMask()
        self.backupImage()

    def destroy(self):
        self.particleRegistrar(Particles(
            position=self.position.copy(),
            life=40, radius=15, rate=30, persistence=5, colors=self.explodeColors
        ))
        self.destructing = True

    def handleSpriteCollision(self, sprite, collisionPoint=None):
        if isinstance(sprite, Asteroid):
            self.destroy()
            print('Missile hit asteroid')
    def handleGroundCollision(self, normal=None):
        self.destroy()
        print('Missile hit ground')

class Asteroid(Sprite):
    def __init__(self, *args, radius=(3, 15), color=[255, 255, 255], **kwargs):
        super().__init__(*args, **kwargs)
        self.x = None
        self.y = None
        self.color = color

        radius = self.handleArgumentRange(radius, 10, 3, 15)

        self.generateCoordinates(radius)
        self.size = [radius*2, radius*2]
        self.generateImage()
        self.updateMask()

    def generateImage(self):
        super().generateImage()
        radius = int(max(self.size)/2)
        x = (self.x + radius).astype('int').tolist()
        y = (self.y + radius).astype('int').tolist()
        pygame.draw.polygon(self.image, self.color, list(zip(x, y)), width=0)
        self.backupImage()

    def generateCoordinates(self, radius):
        numPoints = max([3, int(np.ceil(2*np.pi*radius / 8))])
        angles = np.random.uniform(0, 2*np.pi, size=numPoints)
        angles.sort()
        radii = radius * np.random.uniform(0.6, 1.4, size=numPoints)
        self.x = radii * np.cos(angles)
        self.y = radii * np.sin(angles)

    def handleSpriteCollision(self, sprite, collisionPoint=None):
        if isinstance(sprite, Missile):
            self.destroy()
            print('Asteroid hit missile')

    def handleGroundCollision(self, normal=None):
        self.destroy()
        print('Asteroid hit ground')

class PowerUp(Sprite):
    def __init__(self, *args, powerupType=None, meanSpeed=0.1,
            maxAngularVelocity=2, **kwargs):
        super().__init__(*args, **kwargs)
        speed = np.random.poisson(lam=meanSpeed)
        angle = np.random.uniform(0, np.pi)
        self.angularVelocity = np.random.uniform(-maxAngularVelocity, maxAngularVelocity)
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        self.velocity = np.array([vx, vy], dtype='float')
        self.angularVelocity = np.random.uniform(-0.1, 0.1)
        self.damageRate = 0.1
        self.type = powerupType
        self.captured = False
        self.lander = None
        self.backupImage()
        self.updateMask()

    def handleSpriteCollision(self, sprite, collisionPoint=None):
        self.destroy()
        print('powerup hit sprite of type', type(sprite))

    def handleGroundCollision(self, normal=None):
        self.destroy()
        print('powerup hit ground')

    def destroy(self):
        super().destroy()
        self.kill()

class Pad(Sprite):
    def __init__(self, game, padNumber, width, height, *args, **kwargs):
        super().__init__(game, *args, **kwargs)
        # Position of the pad will be the center of the top surface
        self.width = width
        self.height = height # Note that this is the height of the pad, but the size of the image is slightly larger due to the landing lights
        self.hasLander = False
        self.padNumber = padNumber
        self.padColor = self.game.groundColor
        self.padFontColor = [0, 50, 50]
        self.lightRadius = 8
        self.generateImage()
        self.backupImage()
        self.updateMask()
        self.updateRect()

    def isOnScreen(self):
        if self.rect is None:
            return False
        return self.game.screenRect.colliderect(self.rect)

    def redrawLandingLights(self):
        lightIntensity = int(100 + 155*(np.sin(self.game.time/10)+1)/2)
        if self.hasLander:
            lightColor = [0, lightIntensity, 0]
        else:
            lightColor = [0, 0, lightIntensity]

        # Draw pad lights
        pygame.draw.circle(
            self.image,
            lightColor,
            (self.lightRadius, self.lightRadius-1),
            self.lightRadius,
            width=0,
            draw_top_left=True,
            draw_top_right=True,
        )
        pygame.draw.circle(
            self.image,
            lightColor,
            (self.width - self.lightRadius, self.lightRadius-1),
            self.lightRadius,
            width=0,
            draw_top_left=True,
            draw_top_right=True,
        )

    def drawPadBody(self, surface, color):
        # Draw pad body
        padRect = pygame.Rect((0, self.lightRadius, self.width, self.height))
        pygame.draw.rect(surface, color, padRect)

    def generateImage(self):
        self.image = pygame.Surface((self.width, self.height + self.lightRadius))
        self.image.set_colorkey('black')

        # Draw landing lights
        self.redrawLandingLights()

        # Draw pad body
        self.drawPadBody(self.image, self.padColor)

        # Draw text
        padNumberSurface = self.font.render(
            str(self.padNumber), False, self.padFontColor
        )
        textCenter = (self.image.width/2, self.lightRadius + padNumberSurface.height)
        textRect = padNumberSurface.get_rect(center=textCenter)
        self.image.blit(padNumberSurface, textRect)

    def updateMask(self):
        # Only the pad itself, not the decorations, will go into the mask
        maskImage = self.image.copy()
        maskImage.fill('black')
        maskImage.set_colorkey('black')

        # Draw pad body
        self.drawPadBody(maskImage, 'white')

        self.mask = pygame.mask.from_surface(maskImage)

    def updateRect(self):
        x, y = self.game.world2Screen(*self.position)
        y -= self.lightRadius
        self.rect = self.image.get_rect(midtop=(x, y))

    def update(self):
        if self.isOnScreen():
            self.redrawLandingLights()
        self.updateRect()

    def handleGroundCollision(self):
        pass

class Lander(Sprite):
    def __init__(self, game, *args, **kwargs):
        super().__init__(game, *args, **kwargs)
        self.voidRegistrar = game.registerVoid
        self.thrusterPower = 0.01
        self.maxFuel = 150 # 15
        self.fuel = self.maxFuel
        self.radius = None
        self.exhaustPlume = 0
        self.rotationMatrix = None
        self.radius = None
        self.speed = 0
        self.extraLives = 3
        self.airbags = 1
        self.phaseOuts = 1
        self.lasers = 0
        self.missileCount = 0
        self.createRotationMatrix()
        self.generateImage()
        self.backupImage()
        self.updateMask()
        self.updateRect()
        self.phaseOutTime = 0
        self.phaseOutColors = createColorRange([255, 200, 200], [255, 50, 50], 15)
        self.airbagTime = 0
        self.isOnPad = False
        self.takeoffGrace = 0
        self.lastPad = None

    def fireRCS(self, amount):
        if self.fuel > 0:
            power = self.thrusterPower*12
            # Command fire
            self.angularVelocity += amount * power
            # Auto-stabilize
            self.useFuel(power/3)

    def handleSpriteCollision(self, sprite, collisionPoint=None):
        notes = ''
        if isinstance(sprite, PowerUp):
            # It's a powerup
            self.capturePowerup(sprite)
            notes = 'Got powerup!'
        elif isinstance(sprite, Missile):
            # It's a missile
            pass
        elif isinstance(sprite, Pad):
            if self.isLanded() or self.takeoffGrace > 0 or self.phaseOutTime > 0:
                 pass
            else:
                safe, notes = self.checkSafeLanding()
                if safe:
                    print('lander safely landed on pad')
                    # Shift lander up a bit
                    padTopY = sprite.position[1] - self.radius*0.5
                    self.position[1] = padTopY

                    self.setLandedState(True, pad=sprite)
                else:
                    if self.bouncy:
                        self.bounce()
                    else:
                        print('lander crashed on pad')
                        self.setCrashed()
        else:
            self.setCrashed()
            notes = 'lander hit sprite of type', type(sprite)
            print(notes)
        return notes
    def handleGroundCollision(self, normal=None):
        if self.phaseOutTime > 0:
            return
        elif self.bouncy:
            self.bounce(normal)
        else:
            self.setCrashed()

    def checkSafeLanding(self):
        maxSpeed = 0.35
        maxTilt = 10
        speedGood = self.speed < maxSpeed
        angleGood = abs(self.getOrientation() < maxTilt) or abs(self.getOrientation() - 360) < maxTilt
        if not speedGood and angleGood:
            notes = 'Too fast!'
        elif speedGood and not angleGood:
            notes = 'Not straight'
        elif not speedGood and not angleGood:
            notes = 'Too fast and not straight!'
        else:
            notes = 'Success!'
        return speedGood and angleGood, notes

    def setCrashed(self, showMessage=True):
        self.freeze()
        self.destroy()
        if showMessage:
            self.messageRegistrar(
                Message(
                    text="Crashed!",
                    duration=50,
                    color=[150, 150, 255],
                    color2=[50, 50, 150],
                    flashFrequency=0.1,
                    size=1)
                )
            if self.extraLives > 0:
                self.messageRegistrar(
                    Message(
                        text="Press enter to try again!",
                        duration=150,
                        color=[150, 150, 255],
                        color2=[50, 50, 150],
                        flashFrequency=0.1,
                        size=1)
                    )
            else:
                self.messageRegistrar(
                    Message(
                        text="Game over!",
                        duration=150,
                        color=[150, 150, 255],
                        color2=[50, 50, 150],
                        flashFrequency=0.1,
                        size=1)
                    )
    def setLandedState(self, newLandedState, pad=None, showMessage=True):
        if not self.isLanded() and newLandedState:
            # Lander has just landed
            self.isOnPad = True
            self.freeze()
            self.lastPad = pad
            if showMessage:
                self.messageRegistrar(
                    Message(
                        text="Landed!",
                        duration=100,
                        color=[150, 255, 150],
                        color2=[50, 150, 50],
                        flashFrequency=0.1,
                        size=1))
        elif self.isLanded() and not newLandedState:
            # Lander has just taken off
            self.takeoffGrace = 10
            self.isOnPad = False

    def isLanded(self):
        return self.isOnPad
    def isFlying(self):
        return not self.destructing and self.life > 0 and not self.isLanded()
    def isCrashed(self):
        return self.destructing

    def fireMissile(self):
        velocity = self.velocity.copy()

        newMissile = Missile(
            game,
            thrust=0.1,
            image=MISSILE_IMAGE.copy(),
            orientation=self.getOrientation(),
            position=self.position.copy(),
            velocity=velocity,
            noCollideIDs=[self.ID],
            life=10)

    def fireLaser(self):
        origin = self.position.copy()
        orientation = self.getOrientation()
        width = 40
        length = 600
        void = LaserTunnel(origin, orientation+-90, width, length)
        self.voidRegistrar(void)

    def activatePowerup(self, powerupName):
        # ["Missiles", "Airbags", "PhaseOut"]
        if powerupName == "Missiles":
            if self.missileCount > 0:
                self.fireMissile()
                self.missileCount -= 1
        elif powerupName == "Airbags":
            print("activating airbags!")
        elif powerupName == "PhaseOut":
            maxPhaseOut = 1000
            self.phaseOutTime += maxPhaseOut
            self.particleRegistrar(
                Particles(
                    position=self.position,  # Not a copy, so it follows the sprite
                    life=maxPhaseOut-5, radius=self.radius, rate=30, persistence=5, colors=self.phaseOutColors
                )
            )
        elif powerupName == "Laser":
            if self.lasers > 0:
                self.fireLaser()
                self.lasers -= 1
        else:
            raise ValueError('Unknown powerup {p}'.format(p=powerupName))

    def capturePowerup(self, powerup):
        powerup.life = 0
        if powerup.type == 'Fuel':
            # Fuel just gets used, not stored or activated
            self.fuel = self.maxFuel
        elif powerup.type == 'Missiles':
            self.missileCount += 2
        elif powerup.type == 'ExtraLife':
            self.extraLives += 1
        elif powerup.type == 'Airbags':
            self.airbags = 1
        elif powerup.type == 'PhaseOut':
            self.phaseOuts = 1
        elif powerup.type == 'Laser':
            self.lasers += 1
        else:
            raise ValueError('Unknown powerup type: {type}'.format(type=powerup.type))

    def handlePowerups(self):
        pass

    def update(self, *args, **kwargs):
        super().update(*args, frozen=self.isLanded(), **kwargs)
        if not self.isLanded():
            if not (self.angularVelocity == 0):
                self.incrementOrientation(self.angularVelocity)
                if self.fuel > 0:
                    # If fuel remains, auto-stabilize orientation
                    self.angularVelocity = self.angularVelocity * 0.9
                    if abs(self.angularVelocity) < 0.1:
                        self.angularVelocity = 0
            self.speed = np.sqrt(self.velocity.dot(self.velocity))
            if self.takeoffGrace > 0:
                # Don't worry about crashing or landing if we're just taking off
                self.takeoffGrace -= 0.1

        if self.exhaustPlume > 0:
            self.exhaustPlume = max([0, self.exhaustPlume - 0.5])
        if self.phaseOutTime > 0:
            self.phaseOutTime -= 1

    def freeze(self):
        super().freeze()
        self.speed = 0
        self.exhaustPlume = 0
        self.setOrientation(0)

    def generateImage(self):
        self.image = LANDER_ICON.convert_alpha()
        # # Add 3 color channels to lander image
        # self.image = pygame.Surface(self.image.shape[::-1], pygame.SRCALPHA)

        # scale = 3
        # self.image = pygame.transform.scale_by(self.image, scale)
        self.radius = max(self.image.get_size())
        self.backupImage()

    def setOrientation(self, orientation):
        orientationChanged = super().setOrientation(orientation)
        if orientationChanged:
            self.createRotationMatrix()

    def createRotationMatrix(self):
        self.rotationMatrix = makeRotationMatrix((- self.getOrientation()))

    def drawExhaustPlume(self, screen):
        size = self.exhaustPlume
        if size > 0:
            offset = -self.radius*0.3
            pt0 = (self.position - self.rotationMatrix.dot([-size*0.5, offset]))
            pt1 = (self.position - self.rotationMatrix.dot([0, -(self.radius*0.5 + size) + offset]))
            pt2 = (self.position - self.rotationMatrix.dot([size*0.5, offset]))

            x, y = list(zip(*[pt0, pt1, pt2]))

            x, y = self.game.world2Screen(x, y, integer=True, asList=False)
            pt0, pt1, pt2 = list(zip(x, y))

            pygame.draw.lines(screen, [100, 100, 255], False, [pt0, pt1, pt2], 1)

    def fireThrusters(self):
        if self.fuel > 0 and not self.destructing:
            angle = (270 - self.getOrientation()) * np.pi / 180
            self.velocity += [self.thrusterPower * np.cos(angle), self.thrusterPower * np.sin(angle)]
            self.exhaustPlume = min(self.exhaustPlume+1, self.radius/3)
            self.useFuel(self.thrusterPower)
            self.setLandedState(False)

    def useFuel(self, amount):
        self.fuel = max([0, self.fuel - amount])
        if self.fuel == 0:
            self.messageRegistrar(
                Message(
                    text="Out of fuel!",
                    duration=250,
                    color=[50, 200, 200],
                    color2=[100, 100, 100],
                    flashFrequency=0.1,
                    size=1)
            )

    def destroy(self):
        super().destroy()
        self.exhaustPlume = 0

    def draw(self, screen):
        self.drawExhaustPlume(screen)
        return super().draw(screen)

class Message:
    def __init__(self,
                text='',
                duration=100,
                color=[255, 255, 255],
                color2=None,
                size=0.5,
                flashFrequency=None,
                font=pygame.font.Font(),
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
        self.textSize = self.font.size(self.text)
        self.makeColorPairs()

    def makeColorPairs(self):
        if self.color2 is not None:
            self.colorPairs = list(zip(self.color, self.color2))
        else:
            self.colorPairs = None

    def draw(self, screen, time=0):
        if self.duration > 0:
            w, h = screen.size
            x = int((w - self.textSize[0])/2)
            y = int((h - self.textSize[1])/2)
            if self.flashFrequency is None:
                color = self.color
            else:
                f = (np.sin(time * self.flashFrequency) + 1) / 2
                if self.colorPairs is None:
                    self.makeColorPairs()
                color = [int(c1 * f + c2 * (1-f)) for c1, c2 in self.colorPairs]
            antialias = False
            surface = self.font.render(self.text, antialias, color)
            screen.blit(surface, dest=(x, y))
            self.duration -= 1

class Void:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.xRange = np.array([x.min(), x.max()])
        self.yRange = np.array([y.min(), y.max()])

    def isOnScreen(self, coordinateTransformation, screenWidth, screenHeight):
        xRange, yRange = coordinateTransformation(self.xRange, self.yRange, integer=True)
        xRange = np.array(xRange)
        yRange = np.array(yRange)
        if (xRange > 0).all() and (xRange < screenWidth).all():
            return True
        if (yRange > 0).all() and (yRange < screenHeight).all():
            return True
        if xRange[0] < 0 and xRange[1] > screenWidth:
            return True
        if yRange[0] < 0 and yRange[1] > screenHeight:
            return True
        if np.logical_xor.reduce(xRange > 0):
            return True
        if np.logical_xor.reduce(yRange > 0):
            return True
        if np.logical_xor.reduce(xRange > screenWidth):
            return True
        if np.logical_xor.reduce(yRange > screenHeight):
            return True
        return False

        return xRange[0] >= 0 and \
                xRange[1] <= screenWidth and \
                yRange[0] >= 0 and \
                yRange[1] <= screenHeight

    def draw(self, screens, coordinateTransformation):
        x, y = coordinateTransformation(self.x, self.y, integer=True)
        for screen in screens:
            pygame.draw.polygon(screen, [0, 0, 0], list(zip(x, y)))

class LaserTunnel(Void):
    def __init__(self, origin, orientation, width, length):
        baseVector = length * np.array([np.cos(orientation*np.pi/180), np.sin(orientation*np.pi/180)])
        sideVector = makeRotationMatrix(90).dot(baseVector * ((width/2) / length))
        p0 = origin + sideVector
        p1 = p0 + baseVector
        p2 = p1 - 2 * sideVector
        p3 = p2 - baseVector
        x = np.array([p0[0], p1[0], p2[0], p3[0]])
        y = np.array([p0[1], p1[1], p2[1], p3[1]])
        super().__init__(x, y)

class LunarLanderGame:
    def __init__(self):
        self.w = 1200
        self.h = 800
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((self.w, self.h))
        self.screenRect = self.screen.get_rect()
        self.window_name = 'screen'
        self.screenCenter = np.array([0, 0])
        self.screenScale = 1
        self.groundHeight = 200
        self.maxGroundDepth = 100

        self.ground = pygame.sprite.Sprite()
        self.pads = pygame.sprite.Group()
        self.asteroids = pygame.sprite.Group()
        self.powerups = pygame.sprite.Group()
        self.pads = pygame.sprite.Group()
        self.allEntities = pygame.sprite.Group()

        self.gravity = np.array([0, 0.002], dtype='float')
        self.landerThumbnail = None
        self.landerThumbnailScale = None
        self.landerThumbnailRadius = None
        self.HUDFont = pygame.font.Font()
        self.HUDFontSize = 0.5
        self.HUDColor = [255, 200, 200]
        self.padColor = [150, 150, 100]
        self.groundColor = [155, 155, 155]
        self.time = 0
        self.messages = []
        self.offscreenKillDistance = max([self.h, self.w])*2

        self.selectablePowerups = ["Missiles", "Airbags", "PhaseOut", "Laser"]
        self.selectedPowerup = 0

        self.nextID = 1

        self.voids = []

        self.lander = Lander(self)
        self.allEntities.add(self.lander)

        self.starfield = None

        self.particles = []

        self.paused = False

        self.loadStarfield()

        self.generateLanderThumbnail()

        self.generateGround()
        startingPad = self.chooseStartingPad()
        self.moveToPad(startingPad)
        self.lander.setLandedState(True, pad=startingPad, showMessage=False)
        self.showBeginMessage()

    def pause(self):
        self.paused = True
        # Add paused message
        self.registerMessage(
            Message(
                text="Paused",
                duration=np.inf,
                color=[255, 150, 150],
                color2=[150, 50, 50],
                flashFrequency=0.05,
                size=1)
            )

    def unpause(self):
        self.paused = False
        # Remove paused message
        self.messages = [message for message in self.messages if message.text != 'Paused']

    def togglePause(self):
        if self.isPaused():
            self.unpause()
        else:
            self.pause()

    def isPaused(self):
        return self.paused

    def registerMessage(self, message):
        assert isinstance(message, Message)
        self.messages.append(message)

    def registerParticles(self, particles):
        self.particles.append(particles)

    def registerVoid(self, void):
        self.voids.append(void)

    def loadStarfield(self):
        self.starfield = pygame.image.load(STARFIELD_FILE)

    def generateLanderThumbnail(self):
        self.landerThumbnailScale = 1 #self.lander.radius*2/15
        self.landerThumbnail = pygame.transform.scale_by(self.lander.image, 1/self.landerThumbnailScale)
        self.landerThumbnailRadius = self.lander.radius / self.landerThumbnailScale

    def showBeginMessage(self):
        self.registerMessage(
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
            self.drawStarfield()
            # self.clearScreen()
            if not self.isPaused():
                self.causeEvents()
                self.update()

            visibleRange = self.updateGroundImage()
            self.drawGround()

            # self.drawVoids()
            #
            self.drawSprites()
            self.checkSpriteCollisions()
            #
            self.drawHUD()
            # self.drawParticles()
            self.updateScreenView(visibleRange)

            # Handle key events that should only fire once on keydown
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key not in [pygame.K_RIGHT, pygame.K_LEFT, pygame.K_DOWN, pygame.K_UP]:
                        self.handleKeyPress(event.key)

            # Handle key events that should fire repeatedly as long as key is down
            keys = pygame.key.get_pressed()
            for arrow_key in [pygame.K_RIGHT, pygame.K_LEFT, pygame.K_DOWN, pygame.K_UP]:
                if keys[arrow_key]:
                    self.handleKeyPress(arrow_key)

            pygame.display.flip()
            self.clock.tick(60)

            self.time += 1

    def causeEvents(self):
        c = np.random.uniform()
        if c < 0.0002:
            self.registerMessage(
                Message(
                    text="Asteroid alert!",
                    duration=50,
                    color=[150, 150, 255],
                    color2=[50, 50, 255],
                    flashFrequency=0.1,
                    size=1)
                )
            self.addAsteroids(N=10)
        c = np.random.uniform()
        if c < 0.0015:
            self.createPowerUp()

    def createPowerUp(self, powerupType=None):
        position = self.lander.position + [np.random.uniform(-100, 100), -self.h * self.screenScale + np.random.uniform(-self.h/5, self.h/5)]
        if powerupType is None:
            powerupType = np.random.choice(list(POWERUP_IMAGES.keys()))
        newPowerUp = PowerUp(
            self,
            powerupType=powerupType,
            image=POWERUP_IMAGES[powerupType].copy(),
            position=position)
        self.powerups.add(newPowerUp)
        self.allEntities.add(newPowerUp)

    def addAsteroids(self, N=15):
        for k in range(N):
            position = self.lander.position + [np.random.uniform(-100, 100), -self.h * self.screenScale + np.random.uniform(-self.h/5, self.h/5)]
            newAsteroid = Asteroid(self, position=position, angularVelocity=(-1, 1), radius=(3, 40), speed=1.5, color=self.groundColor)
            self.asteroids.add(newAsteroid)
            self.allEntities.add(newAsteroid)

    def generateGround(self, nPads=10, nChasms=10, nMountains=3, nCaves=25):

        # debug
        # nChasms = 0
        # nMountains = 0
        # nPads = 2
        # nCaves = 0

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
        minW = 2
        maxW = 20
        minExtrusions = 20
        maxExtrusions = 70
        minSubCaves = 1
        maxSubCaves = 15
        for k in range(nCaves):
            N = self.groundX.size
            # Choose some ground segments to extrude
            idx0 = maxW
            idx1 = N - maxW
            nSubCaves = np.random.randint(minSubCaves, maxSubCaves)
            for subCave in range(nSubCaves):
                # Choose a cave center idx between idx0 and idx1
                centerIdx = np.random.randint(idx0, idx1)
                w = np.random.randint(minW, maxW)
                # create new idx0 and idx1 around center
                idx0 = max([centerIdx - int(w/2), 0])
                idx1 = min([centerIdx + int(w/2), N])
                nExtrusions = np.random.randint(minExtrusions, maxExtrusions)
                # Get the idx0 and new idx1 of the cave
                # idx1 = self.addCaveElement(idx0, idx1, addCavePassage, nExtrusions=nExtrusions, maxAngularDeviation=6)
                idx1 = self.addCaveElement(idx0, idx1, addCaveChamber, radius=150)

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

        padSize = int(self.lander.radius*5 / meanDeltaX)
        for k, padIdx in enumerate(padIndices):
            x0 = self.groundX[padIdx-padSize:padIdx+padSize].min()
            x1 = self.groundX[padIdx-padSize:padIdx+padSize].max()
            y1 = self.groundY[padIdx-padSize:padIdx+padSize].min()
            y0 = self.groundY[padIdx-padSize:padIdx+padSize].max() + 1
            position = ((x0 + x1)/2, y1)
            newPad = Pad(
                self,
                k,
                x1-x0,
                y0-y1,
                font='arial',
                position=position
            )
            self.pads.add(newPad)
            self.allEntities.add(newPad)

    def addCaveElement(self, idx0, idx1, algorithm, *args, nTrys=5, **kwargs):
        positionLeft  = np.array([self.groundX[idx0], self.groundY[idx0]])
        positionRight = np.array([self.groundX[idx1], self.groundY[idx1]])
        intersects = False
        for tries in range(nTrys):
            # print('Attempt to make cave element, try #', tries)
            insertX, insertY = algorithm(positionLeft, positionRight, *args, **kwargs)

            # Check if it intersects
            intersects = findPolylineIntersection(self.groundX, self.groundY, insertX, insertY)
            if intersects:
                pass
                # print('Found intersection, trying again')
            else:
                # print('No intersections!')
                break

        if intersects:
            return idx1
        else:
            self.groundX = np.array(self.groundX[:idx0].tolist() + insertX + self.groundX[idx1:].tolist())
            self.groundY = np.array(self.groundY[:idx0].tolist() + insertY + self.groundY[idx1:].tolist())
            newIdx1 = idx1 + len(insertX)
            return newIdx1

    def chooseStartingPad(self):
        padIdx = int(np.round(len(self.pads)/2))
        return self.pads.sprites()[padIdx]

    def moveToPad(self, pad):
        padCenterX = pad.position[0]
        padTopY = pad.position[1] - self.lander.radius*0.5
        self.lander.position = np.array([padCenterX, padTopY], dtype='float')
        self.lander.setOrientation(0)
        self.lander.angularVelocity = 0
        self.screenCenter[0] = padCenterX
        self.lander.updateRect()

    def getClosestPad(self, x, y):
        # Get index of the closest pad to the coordinates
        # World x and world y, not screen
        padXs = (self.padX0s + self.padX1s) / 2
        padYs = self.padY1s
        taxicabDistances = abs(x - padXs) + abs(y - padYs)
        padIdx = np.argmin(taxicabDistances)
        return self.pads.get_sprite(padIdx)

    def isSpriteOutOfBounds(self, sprite):
        x, y = self.world2Screen(*sprite.position)
        return x < -self.offscreenKillDistance or x > self.offscreenKillDistance or y < -self.offscreenKillDistance or y > self.offscreenKillDistance

    def drawSprites(self):
        self.lander.drawExhaustPlume(self.screen)
        self.allEntities.draw(self.screen)

    def checkSpriteCollisions(self):
        # Check asteroid/powerup collision - powerup dies
        sprites = self.allEntities.sprites()
        for k, sprite1 in enumerate(sprites):
            for sprite2 in sprites[k+1:]:
                collisionPoint = pygame.sprite.collide_mask(sprite1, sprite2)
                if collisionPoint is not None:
                    sprite1.handleSpriteCollision(sprite2, collisionPoint=collisionPoint)
                    sprite2.handleSpriteCollision(sprite1, collisionPoint=collisionPoint)

        # Check sprite/ground collision
        crashed_entities = pygame.sprite.spritecollide(self.ground, self.allEntities, False, collided=pygame.sprite.collide_mask)
        for sprite in crashed_entities:
            sprite.handleGroundCollision()

    def drawParticles(self):
        deadParticles = []
        for k, particle in enumerate(self.particles):
            if particle.isAlive():
                particle.draw(self.screen, coordinateTransform=self.world2Screen)
            else:
                deadParticles.append(k)
        if len(deadParticles) > 0:
            self.particles = [particle for k, particle in enumerate(self.particles) if k not in deadParticles]

    def drawMessage(self):
        if len(self.messages) > 0:
            self.messages[0].draw(self.screen, time=self.time)
            if self.messages[0].duration <= 0:
                self.messages.pop(0)

    def drawHUD(self):
        wHUD = int(0.2 * self.w)
        hHUD = int(0.2 * self.h)
        # HUD outline
        pygame.draw.rect(self.screen, self.HUDColor, (0, 0, wHUD, hHUD), 1)

        spacing = int(hHUD*0.01)
        x_left = int(wHUD*0.05)

        # Fuel gauge label
        surface = self.HUDFont.render('Fuel', False, self.HUDColor)
        x = x_left
        y = 15
        self.screen.blit(surface, (x, y))
        y = y + surface.height + spacing
        wFuel = int(wHUD*0.75)
        hFuel = int(hHUD*0.05)
        fFuel = self.lander.fuel / self.lander.maxFuel
        wFuelRemaining = int(wFuel * fFuel)

        # Fuel level
        if fFuel < 0.15:
            fuelColor = [0, 0, 255]
        elif fFuel < 0.3:
            fuelColor = [0, 255, 255]
        else:
            fuelColor = [0, 255, 0]

        pygame.draw.rect(
            self.screen,
            fuelColor,
            (x, y, wFuelRemaining, hFuel)
            )
        # Fuel outline
        pygame.draw.rect(
            self.screen,
            self.HUDColor,
            (x, y, wFuel, hFuel),
            1
            )

        # Velocity readout
        y = y + hFuel + spacing
        antialias = False
        surface = self.HUDFont.render(
            'Speed:  {s:.02f} m/s'.format(s=self.lander.speed),
            antialias,
            self.HUDColor)
        self.screen.blit(surface, dest=(x, y))
        y = y + surface.height + spacing

        # Status readout
        if self.lander.isLanded():
            status='Landed'
        elif self.lander.isCrashed():
            status = 'Crashed'
        else:
            status = 'Flying'

        antialias = False
        surface = self.HUDFont.render(
            'Status: {status}'.format(status=status),
            antialias,
            self.HUDColor)
        self.screen.blit(surface, dest=(10, 65))
        y = y + surface.height + spacing

        # Draw extra lives
        self.screen.blit(self.landerThumbnail, dest=(x, y))
        # y += int(2*self.landerThumbnailRadius) + spacing

        antialias = False
        landerRect = self.landerThumbnail.get_rect(topleft=(x, y))
        self.screen.blit(self.landerThumbnail, dest=landerRect)
        surface = self.HUDFont.render(
            'x{lives}'.format(lives=self.lander.extraLives),
            antialias,
            self.HUDColor)
        textRect = surface.get_rect(midleft=landerRect.midright)
        self.screen.blit(surface, dest=textRect)

        # Draw missiles
        x = x + textRect.midright[0] + spacing
        icon = self.getIcon(MISSILE_ICON, self.lander.missileCount > 0)
        y = 79 - int(icon.size[0]/2)
        x0 = x - 5
        y0 = y
        y1 = y + icon.size[1]
        drawImage(icon, self.screen, position=np.array((x, y)), worldUnits=False)
        x += 10 + icon.size[0] // 2

        # antialias = False
        # surface = self.HUDFont.render(
        #     'x{missiles}'.format(missiles=self.lander.missileCount),
        #     antialias,
        #     self.HUDColor)
        # self.screen.blit(surface, dest=(x, 85))
        #
        # # Draw airbag
        # x += 10 + MISSILE_ICON.size[0]
        # icon = self.getIcon(AIRBAGS_ICON, self.lander.airbags > 0)
        # x1 = x - 5
        # drawImage(icon, self.screen, position=np.array((x, 79 - int(AIRBAGS_ICON.size[0]/2))), worldUnits=False)
        # # Draw phase out
        # x += 5 + icon.size[0]
        # icon = self.getIcon(PHASEOUT_ICON, self.lander.phaseOuts > 0)
        # x2 = x - 5
        # drawImage(icon, self.screen, position=np.array((x, 79 - int(icon.size[0]/2))), worldUnits=False)
        # x3 = x2 + icon.size[0] + 5
        # # Draw phase out
        # x += 5 + icon.size[0]
        # icon = self.getIcon(LASER_ICON, self.lander.lasers > 0)
        # x3 = x - 5
        # drawImage(icon, self.screen, position=np.array((x, 79 - int(icon.size[0]/2))), worldUnits=False)
        # x4 = x3 + icon.size[0] + 5
        #
        # # Draw select box:
        # ybox = [y0, y1]
        # if self.selectedPowerup == 0:
        #     p0 = (x0, y0);
        #     size = (x1 - x0, y1 - y0);
        # elif self.selectedPowerup == 1:
        #     p0 = (x1, y0);
        #     size = (x2 - x1, y1 - y0);
        # elif self.selectedPowerup == 2:
        #     p0 = (x2, y0);
        #     size = (x3 - x2, y1 - y0);
        # elif self.selectedPowerup == 3:  # Laser
        #     p0 = (x3, y0);
        #     size = (x4 - x3, y1 - y0);
        # pygame.draw.rect(self.screen, self.HUDColor, p0 + size, 1)

    def getIcon(self, icon, dimmed):
        if dimmed:
            return icon
        else:
            # Dim out icon
            originalSize = icon.size
            icon_array = np.zeros(originalSize + (3,), dtype='uint8')
            pygame.pixelcopy.surface_to_array(icon_array, icon)
            icon_array = icon_array // 2
            pygame.pixelcopy.array_to_surface(icon, icon_array)
            return icon

    def useExtraLife(self):
        if self.lander.extraLives > 0:
            self.lander.extraLives -= 1
            self.lander.fuel = self.lander.maxFuel
            self.lander.resetImage()
            self.lander.resetState()
            self.moveToPad(self.lander.lastPad)
            self.lander.setLandedState(True, pad=self.lander.lastPad, showMessage=False)
            self.showBeginMessage()

    def handleKeyPress(self, key):
        numSelectablePowerups = len(self.selectablePowerups)

        if key == pygame.K_RETURN:
            if self.lander.life <= 0:
                # Use extra life and reset
                if self.lander.extraLives > 0:
                    self.useExtraLife()
                else:
                    pygame.quit()
        if key == pygame.K_ESCAPE:
            self.togglePause()
        if key == pygame.K_z:
            self.selectedPowerup = (self.selectedPowerup - 1) % numSelectablePowerups
        if key == pygame.K_x:
            self.selectedPowerup = (self.selectedPowerup + 1) % numSelectablePowerups
        if key == pygame.K_SPACE:
            self.lander.activatePowerup(self.selectablePowerups[self.selectedPowerup])
        if key == pygame.K_q:
            pygame.quit()
        if not self.lander.isCrashed() and not self.isPaused():
            if key == pygame.K_UP:
                self.lander.fireThrusters()
                if self.lander.isLanded():
                    self.lander.takeoffGrace = 10
                    # Extra kick
                    self.lander.fireThrusters()
            if key == pygame.K_a:
                # self.addAsteroids()
                # self.createPowerUp(powerupType='Laser')
                # self.particles.append(Particles(
                #     position=self.lander.position.copy(),
                #     life=100, radius=50, rate=5, persistence=20, colors=np.ones([1, 3])*255
                # ))
                # self.fireMissile()
                # self.lander.activatePowerup('PhaseOut')
                self.lander.fireLaser()

            if self.lander.isLanded():
                # Don't respond to other keys if landed
                return

            if key == pygame.K_LEFT:
                self.lander.fireRCS(1)
            if key == pygame.K_DOWN:
                pass
            if key == pygame.K_RIGHT:
                self.lander.fireRCS(-1)

    def update(self):
        if self.lander.isLanded():
            # Refuel
            self.lander.fuel = min([self.lander.maxFuel, self.lander.fuel + 0.2])

        self.allEntities.update()

    def clearScreen(self):
        self.screen.fill([0, 0, 0])

    def drawStarfield(self):
        backgroundParallaxRate = 4
        x, y = (self.screenCenter / backgroundParallaxRate).astype('int').tolist()
        w, h = self.starfield.size
        startX = x - ((x // w) + 1) * w
        nX = (x + self.w - startX) // w + 1
        startY = y - ((y // h) + 1) * h
        nY = (y + self.h - startY) // h + 1
        for kx in range(nX):
            for ky in range(nY):
                self.screen.blit(self.starfield, dest=(startX + kx * w, startY + ky * h))

    def updateScreenView(self, visibleRange):
        x, y = self.world2Screen(*self.lander.position)
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

        zoomSpeed = 1.001
        minScreenScale = 1

        output = False
        if distFromTopMargin < 0: # and groundBotMargin > 0:
            # Shift view up
            # print('shifting up')
            self.screenCenter[1] += distFromTopMargin/10
        elif distFromBotMargin < 0: #and (visibleGroundGoodMargin < 0 or distFromBotMargin < 0):
            # Shift view down
            # print('shifting down')
            self.screenCenter[1] -= distFromBotMargin/10

        if groundBotMargin < 0:
            # Zoom out
            # print('zooming out')
            self.screenScale *= zoomSpeed
        elif self.screenScale > minScreenScale: #distFromTopMargin >= 0 and groundBotMargin >= 0 and self.screenScale > minScreenScale:
            # zoom in
            # print('zooming in')
            self.screenScale /= zoomSpeed
            self.screenCenter[1] += 1

        # Limit zoom-in to 1
        self.screenScale = max([minScreenScale, self.screenScale])

    def world2Screen(self, worldX, worldY, integer=False, asList=False, checkValid=False):
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
        if checkValid:
            valid = [x >= 0 and x < self.screen.shape[1] and y >= 0 and y < self.screen.shape[0] for x, y in zip(screenX, screenY)]
            return screenX, screenY, valid
        else:
            return screenX, screenY

    def updateGroundImage(self):
        if self.ground.image is None:
            self.ground.image = pygame.Surface((self.w, self.h))
            self.ground.rect = pygame.Rect(0, 0, self.w, self.h)
        else:
            self.ground.image.fill([0, 0, 0])

        groundX, groundY = self.world2Screen(self.groundX, self.groundY)
        onScreen = np.logical_and(groundX >= 0, groundX <= self.w)
        onScreenIdx = np.nonzero(onScreen)[0]
        if sum(onScreen) > 0:
            visibleRange = [groundY[onScreen].max(), groundY[onScreen].min()]
            firstIdx = max([1, onScreenIdx.min()-1])
            lastIdx = min([len(groundX), onScreenIdx.max()+1])
            groundX = np.append(groundX[firstIdx:lastIdx+1], [groundX[lastIdx+1], groundX[firstIdx]])
            groundY = np.append(groundY[firstIdx:lastIdx+1], [self.h*2, self.h*2])
        else:
            visibleRange = [self.h, self.h]
        rect = pygame.draw.polygon(self.ground.image, self.groundColor, list(zip(groundX, groundY)))

        self.ground.image.set_colorkey([0, 0, 0])
        self.ground.mask = pygame.mask.from_surface(self.ground.image)
        return visibleRange

    def drawGround(self):
        self.screen.blit(self.ground.image)

    def drawVoids(self):
        for void in self.voids:
            if void.isOnScreen(self.world2Screen, self.w, self.h):
                void.draw([self.screen, self.collisionMap], self.world2Screen)

def drawImage(image, screen, scale=1, coordinateTransform=None, worldUnits=True,
        position=np.zeros(2, dtype='float'), orientation=0):

    if worldUnits and coordinateTransform is not None:
        x, y = coordinateTransform(*position)
    else:
        x, y = position
    image = getRotatedAndScaledImage(image, orientation, scale)
    rect = image.get_rect(center=(x, y))

    screen.blit(image, rect)

def getRotatedAndScaledImage(image, angle=0, scale=1, centered=False):
        if angle == 0 and scale == 1:
            return image.copy()
        else:
            return pygame.transform.rotozoom(image, angle, scale)


def createColorRange(color1, color2, nSteps):
    colors = np.zeros([nSteps, 3], dtype='uint8')
    stepSize = (np.array(color2) - np.array(color1)) / nSteps
    for channel in range(3):
        if stepSize[channel] != 0:
            colors[:, channel] = np.arange(color1[channel], color2[channel], stepSize[channel])
        else:
            colors[:, channel] = np.ones(nSteps)*color1[channel]
    return colors

def generateRandomVelocity(speedRange, angleRange=None):
    try:
        startSpeed = speedRange[0]
        endSpeed = speedRange[1]
    except TypeError:
        startSpeed = 0
        endSpeed = speedRange
    if angleRange is None:
        angleRange = (0, 2*np.pi)

    speed = np.random.uniform(startSpeed, endSpeed)
    angle = np.random.uniform(*angleRange)
    velocity = speed * np.array([np.cos(angle), np.sin(angle)], dtype='float')
    return velocity

def disintegrateImage(image, damageCount=50):
    rng = np.random.default_rng()
    radius_x = int(image.width/2)
    radius_y = int(image.height/2)
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

    out_of_range = np.logical_or.reduce([x_end < 0, x_end >= image.width, y_end < 0, y_end >= image.height])

    imarray = pygame.surfarray.pixels3d(image)
    vals = imarray[x_start, y_start, :]

    vals = np.delete(vals, out_of_range, axis=0)
    x_end = np.delete(x_end, out_of_range)
    y_end = np.delete(y_end, out_of_range)

    # Copy pixels
    imarray[x_end, y_end, :] = vals
    # Clear pixels
    imarray[x_start, y_start, :] = [0, 0, 0]

def addCaveChamber(positionLeft, positionRight, radius=100, rot90=makeRotationMatrix(90)):
    delta = (positionRight - positionLeft) / np.linalg.norm(positionLeft - positionRight)
    normal = rot90.dot(delta)
    openingCenter = (positionRight + positionLeft)/2
    throatLength = radius * 0.75
    center = openingCenter + normal * (radius + throatLength)
    centerX = center[0]
    centerY = center[1]
    # Make circle
    pointsX = []
    pointsY = []
    maxRandom = radius / 5
    angleStep = 2*np.pi*radius/40
    radiusX = radius * np.random.uniform(0.8, 1.2)
    radiusY = radius * np.random.uniform(0.8, 1.2)
    for a in np.arange(270 + 360 - angleStep, 270 + angleStep, -angleStep):
        newX = centerX + radiusX * np.cos(a*np.pi/180) + np.random.uniform(0, maxRandom)
        newY = centerY + radiusY * np.sin(a*np.pi/180) + np.random.uniform(0, maxRandom)
        pointsX.append(newX)
        pointsY.append(newY)
    return pointsX, pointsY

def addCavePassage(positionLeft, positionRight, nExtrusions=35, rot90=makeRotationMatrix(90), maxAngularDeviation=6):
    extrusionLength = 10;
    delta = extrusionLength * (positionRight - positionLeft) / np.linalg.norm(positionLeft - positionRight)
    normalLeft = rot90.dot(delta)
    normalRight = normalLeft.copy()
    xInsertLeft = [positionLeft[0]]
    xInsertRight = [positionRight[0]]
    yInsertLeft = [positionLeft[1]]
    yInsertRight = [positionRight[1]]
    for j in range(nExtrusions):
        angleLeft, angleRight = np.random.uniform(-maxAngularDeviation, maxAngularDeviation, size=2)
        deviateLeft = makeRotationMatrix(angleLeft)
        deviateRight = makeRotationMatrix(angleRight)
        normalLeft = deviateLeft.dot(normalLeft)
        normalRight = deviateRight.dot(normalRight)
        nextLeft = [xInsertLeft[-1], yInsertLeft[-1]] + normalLeft
        nextRight = [xInsertRight[-1], yInsertRight[-1]] + normalRight
        xInsertLeft.append(nextLeft[0])
        yInsertLeft.append(nextLeft[1])
        xInsertRight.append(nextRight[0])
        yInsertRight.append(nextRight[1])
    return xInsertLeft + xInsertRight[::-1], yInsertLeft + yInsertRight[::-1]

def findPolylineIntersection(polyX, polyY, poly2X, poly2Y):
    polyX = np.stack((polyX[:-1], polyX[1:]), axis=0)
    polyY = np.stack((polyY[:-1], polyY[1:]), axis=0)

    poly2X = np.stack((poly2X[:-1], poly2X[1:]), axis=0)
    poly2Y = np.stack((poly2Y[:-1], poly2Y[1:]), axis=0)

    # Cramer's rule
    a1 = polyY[0, :] - polyY[1, :]
    a2 = poly2Y[0, :] - poly2Y[1, :]
    b1 = polyX[1, :] - polyX[0, :]
    b2 = poly2X[1, :] - poly2X[0, :]

    denominator = a1[:, None] * b2[None, :] - b1[:, None] * a2[None, :]

    c1 = polyX[1, :] * polyY[0, :] - polyX[0, :] * polyY[1, :]
    c2 = poly2X[1, :] * poly2Y[0, :] - poly2X[0, :] * poly2Y[1, :]

    x = (c1[:, None] * b2[None] - b1[:, None] * c2[None]) / denominator
    y = (a1[:, None] * c2[None] - c1[:, None] * a2[None]) / denominator

    intersect = np.array([x, y])

    # Filter for intersections on segment
    polyDX = np.diff(polyX, axis=0)
    polyDY = np.diff(polyY, axis=0)
    poly2DX = np.diff(poly2X, axis=0)
    poly2DY = np.diff(poly2Y, axis=0)
    polyDVec = np.stack([polyDX, polyDY]).squeeze()
    poly2DVec = np.stack([poly2DX, poly2DY]).squeeze()
    polyVec = np.array([polyX[0, :], polyY[0, :]])
    poly2Vec = np.array([poly2X[0, :], poly2Y[0, :]])

    polyTointersect = intersect - polyVec[..., None]
    poly2Tointersect = intersect - poly2Vec[:, None, :]

    intersectOnPoly = np.linalg.norm(polyTointersect, axis=0) / np.linalg.norm(polyDVec[..., None], axis=0) <= 1
    intersectOnPoly2 = np.linalg.norm(poly2Tointersect, axis=0) / np.linalg.norm(poly2DVec[:, None, :], axis=0) <= 1

    intersect = (intersectOnPoly & intersectOnPoly2).any()
    # print('INTERSECT?!?!?!', intersect)

    return intersect

root = Path(__file__).parent

ASSET_DIR = root / 'assets'

POWERUP_TYPES = ['Airbags', 'ExtraLife', 'Fuel', 'Missiles', 'PhaseOut', 'Laser']
POWERUP_FILES = dict((powerup_type, str(ASSET_DIR / 'PowerUp_{name}.png'.format(name=powerup_type))) for powerup_type in POWERUP_TYPES)

# Load images
POWERUP_IMAGES = {}
for powerup_type in POWERUP_TYPES:
    POWERUP_IMAGES[powerup_type] = pygame.image.load(POWERUP_FILES[powerup_type])
    POWERUP_IMAGES[powerup_type].set_colorkey('black')
# POWERUP_ATTRIBUTES = dict((powerup_type, {}) for powerup_type in POWERUP_TYPES)
# POWERUP_ATTRIBUTES['Fuel']['storeable'] = False
# POWERUP_ATTRIBUTES['Missiles']['storeable'] = False
STARFIELD_FILE = ASSET_DIR / 'Starfield.png'

MISSILE_IMAGE = pygame.image.load(ASSET_DIR / 'MissileSprite.png')
MISSILE_ICON = cropImage(getRotatedAndScaledImage(pygame.image.load(ASSET_DIR / 'MissileIcon.png'), scale=1.25))
AIRBAGS_ICON = cropImage(getRotatedAndScaledImage(pygame.image.load(ASSET_DIR / 'AirbagsIcon.png'), scale=1.5))
PHASEOUT_ICON = cropImage(getRotatedAndScaledImage(pygame.image.load(ASSET_DIR / 'PhaseOutIcon.png'), scale=1.5))
LASER_ICON = cropImage(getRotatedAndScaledImage(pygame.image.load(ASSET_DIR / 'LaserIcon.png'), scale=1.5))
LANDER_ICON = pygame.image.load(ASSET_DIR / 'Lander.png')

if __name__ == '__main__':

    llg = LunarLanderGame()

    llg.run()
