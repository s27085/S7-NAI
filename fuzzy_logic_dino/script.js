const { Logic } = await import("https://esm.run/es6-fuzz");
const logic = new Logic();
const { Trapezoid, Triangle, Shape, Grade, ReverseGrade } = logic.c;

const logBox = document.createElement("div");
logBox.style.position = "fixed";
logBox.style.bottom = "20px";
document.body.appendChild(logBox);

const distLogic = new Logic()
  .init("veryClose", new ReverseGrade(0, 120))
  .or("close", new Triangle(60, 100, 140))
  .or("far", new Grade(120, 220));

const speedLogic = new Logic()
  .init("slow", new Trapezoid(0, 0, 6, 8))
  .or("medium", new Triangle(7, 9.5, 12))
  .or("fast", new Grade(11, 14));

const gapLogic = new Logic()
  .init("short", new Trapezoid(0, 0, 50, 100))
  .or("medium", new Triangle(80, 150, 250))
  .or("long", new Grade(250, 800));

const game = () => {
  const tRexPos = runnerInstance.tRex.xPos;
  const nextObstacle = runnerInstance.horizon.obstacles[0];
  const nextObstaclePos = nextObstacle?.xPos || 9999999;
  const jump = () => runnerInstance.tRex.startJump(100);

  const dist = nextObstaclePos - tRexPos;
  const gap = nextObstacle?.gap;
  const speed = runnerInstance.currentSpeed;

  const defuzzifiedDist = distLogic.defuzzify(dist).defuzzified;
  const defuzzifiedGap = gapLogic.defuzzify(gap).defuzzified;
  const defuzzifiedSpeed = speedLogic.defuzzify(speed).defuzzified;

  logBox.innerText = `Distance: ${defuzzifiedDist} | Gap: ${defuzzifiedGap} | Speed: ${defuzzifiedSpeed}`;

  let action = "idle";

  if (defuzzifiedDist === "veryClose") {
    action = "jump";
  } else if (defuzzifiedDist === "close" && defuzzifiedSpeed !== "slow") {
    action = "jump";
  }

  if (action === "jump") {
    jump();
  }
};

setInterval(game, 10);
