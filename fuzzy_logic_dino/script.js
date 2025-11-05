//Objective: Create a fuzzy logic controller to play the Chrome Dino game on its own, without human input.
//Authors: Fabian Fetter, Konrad Fijałkowski
//How to run: Open the Chrome Dino game in Google Chrome. Open the DevTools (F12) and paste this script into the console or create a snippet in sources. Press Enter to start the program.

//Import the fuzzy logic library from a CDN
const { Logic } = await import("https://esm.run/es6-fuzz");
const logic = new Logic();
const { Trapezoid, Triangle, Shape, Grade, ReverseGrade } = logic.c;


// Create a log box to display fuzzy logic values at the bottom of the screen
const logBox = document.createElement("div");
logBox.style.position = "fixed";
logBox.style.bottom = "20px";
document.body.appendChild(logBox);


// Define fuzzy logic for distance, speed, obstacle height, and action using fuzzy sets
const distLogic = new Logic()
  .init("veryClose", new ReverseGrade(0, 80))
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

const actionLogic = new Logic()
  .init("idle", new ReverseGrade(0, 0.3))
  .or("duck", new Triangle(0.2, 0.5, 0.7))
  .or("jump", new Grade(0.6, 1));


// Access the global runner instance of the Dino game - actual game logic
const game = () => {
  
  const tRexPos = runnerInstance.tRex.xPos;
  const nextObstacle = runnerInstance.horizon.obstacles[0];
  const nextObstaclePos = nextObstacle?.xPos || 9999999;
  const jump = () => runnerInstance.tRex.startJump(100);

  const dist = nextObstaclePos - tRexPos;
  const obstacleHeight = runnerInstance.horizon.dimensions.height - nextObstacle?.yPos;
  const speed = runnerInstance.currentSpeed;

  //Fuzzification of inputs obtained from the game state
  const defuzzifiedDist = distLogic.defuzzify(dist).defuzzified;
  const defuzzifiedObstacleHeight =
    obstacleHeightLogic.defuzzify(obstacleHeight).defuzzified;
  const defuzzifiedSpeed = speedLogic.defuzzify(speed).defuzzified;

  //Display fuzzy logic values in the log box
  logBox.innerText = `Distance: ${defuzzifiedDist} (${dist}) | Obstacle Height: ${defuzzifiedObstacleHeight} (${obstacleHeight}) | Speed: ${defuzzifiedSpeed} (${speed})`;

  //Rule evaluation to determine action based on fuzzified inputs
  let action = "idle";

  if (defuzzifiedDist === "veryClose") {
    action = defuzzifiedObstacleHeight === "low" ? "jump" : "duck";
  } else if (
    defuzzifiedDist === "close" &&
    defuzzifiedObstacleHeight === "low" &&
    defuzzifiedSpeed !== "slow"
  ) {
    action = "jump";
  }
  //Defuzzification and action execution based on the determined action
  if (action === "jump") {
    jump();
  } else if (action === "duck" && !runnerInstance.tRex.ducking) {
    runnerInstance.tRex.setDuck(true);
    setTimeout(() => {
      runnerInstance.tRex.setDuck(false);
    }, 200);
  }
};

//Run the game loop at regular intervals
setInterval(game, 10);
