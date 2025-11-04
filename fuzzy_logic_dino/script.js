const { Logic } = await import("https://esm.run/es6-fuzz");
const logic = new Logic();
const { Trapezoid, Triangle, Shape, Grade, ReverseGrade } = logic.c;

const logBox = document.createElement("div");
logBox.style.position = "fixed";
logBox.style.bottom = "20px";
document.body.appendChild(logBox);

const distLogic = new Logic()
  .init("veryClose", new ReverseGrade(0, 70))
  .or("close", new Triangle(70, 90, 100))
  .or("far", new Grade(100, 200));

const speedLogic = new Logic()
  .init("slow", new ReverseGrade(0, 8))
  .or("medium", new Triangle(7, 9.5, 12))
  .or("fast", new Grade(11, 14));

const obstacleHeightLogic = new Logic()
  .init("low", new ReverseGrade(0, 60))
  .or("medium", new Triangle(60, 70, 80))
  .or("high", new Grade(80, 150));

const game = () => {
  const tRexPos = runnerInstance.tRex.xPos;
  const nextObstacle = runnerInstance.horizon.obstacles[0];
  const nextObstaclePos = nextObstacle?.xPos || 9999999;
  const jump = () => runnerInstance.tRex.startJump(100);

  const dist = nextObstaclePos - tRexPos;
  const obstacleHeight =
    runnerInstance.horizon.dimensions.height - nextObstacle?.yPos;
  const speed = runnerInstance.currentSpeed;

  const defuzzifiedDist = distLogic.defuzzify(dist).defuzzified;
  const defuzzifiedObstacleHeight =
    obstacleHeightLogic.defuzzify(obstacleHeight).defuzzified;
  const defuzzifiedSpeed = speedLogic.defuzzify(speed).defuzzified;

  logBox.innerText = `Distance: ${defuzzifiedDist} (${dist}) | Obstacle Height: ${defuzzifiedObstacleHeight} (${obstacleHeight}) | Speed: ${defuzzifiedSpeed} (${speed})`;

  let action = "idle";
  //Rules
  if (defuzzifiedDist === "veryClose") {
    action = defuzzifiedObstacleHeight === "low" ? "jump" : "duck";
  } else if (
    defuzzifiedDist === "close" &&
    defuzzifiedObstacleHeight === "low" &&
    defuzzifiedSpeed !== "slow"
  ) {
    action = "jump";
  }

  if (action === "jump") {
    jump();
  } else if (action === "duck" && !runnerInstance.tRex.ducking) {
    runnerInstance.tRex.setDuck(true);
    setTimeout(() => {
      runnerInstance.tRex.setDuck(false);
    }, 200);
  }
};

setInterval(game, 10);
