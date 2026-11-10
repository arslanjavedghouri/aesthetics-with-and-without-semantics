/******************************************************* 
 * Human Evaluation Of Image Similarity Judgments *
 *******************************************************/

import { core, data, sound, util, visual, hardware } from './lib/psychojs-2024.2.4.js';
const { PsychoJS } = core;
const { TrialHandler, MultiStairHandler } = data;
const { Scheduler } = util;
//some handy aliases as in the psychopy scripts;
const { abs, sin, cos, PI: pi, sqrt } = Math;
const { round } = util;


// store info about the experiment session:
let expName = 'Human Evaluation of Image Similarity Judgments';  // from the Builder filename that created this script
let expInfo = {
    'participant': `${util.pad(Number.parseFloat(util.randint(0, 999999)).toFixed(0), 6)}`,
    'age': '',
    'gender': ["", "Male", "Female", "Other"],
    'country of origin': '',
    'do you have any photography or visual art experience?': ["", "Yes", "No"],
};

// Start code blocks for 'Before Experiment'
///////////////////////////
// "Before Experiment" TAB
///////////////////////////

// Declare globals
var CSV_FILE           = 'csv/stimuli.csv';
var ORIGINAL_FOLDER    = 'images/originals_needed/';
var GENERATED_FOLDER   = 'images/generated/';

var CUSTOM_BINS        = [0.000,0.300,0.400,0.457,0.514,0.571,0.629,0.686,0.743,0.800,1.000];
var BIN_LABELS         = CUSTOM_BINS.slice(0,-1)
                            .map((b,i) => b.toFixed(2) + '-' + CUSTOM_BINS[i+1].toFixed(2));

var NUM_TRIALS         = 282;
var NUM_REVERSALS      = 6;
var MIN_TRIALS         = 60;
var STEPSIZE           = 1;

// **NEW: trial‐by‐trial state**  
var image1_path    = "";  
var image2_path    = "";  
var info_str       = "";  
var currentRow     = null;  

// Dataframe and bin structures
var df                  = [];
var possibleValues      = [];
var binsToIndices       = {};
var imgOrderPerBin      = {};
var usedIdxInBin        = {};
var usedGlobalIndices   = new Set();

// Staircase containers
var staircases, whichStairIndex;
// NEW: record which staircase we used each trial
var stairIndices = [];
// Logging arrays
var history       = [];
var responses     = [];
var reaction_times= [];
var key_presses   = [];
var orig_lefts    = [];
var shownRows     = [];

// Utility: CSV parser

function parseCSV(text) {
    let lines   = text.trim().split(/\r?\n/);
    let headers = lines.shift().split(',');
    return lines.map(line => {
        let cols = line.split(',');
        let obj  = {};
        headers.forEach((h,i) => obj[h] = cols[i]);
        obj.clip_similarity = parseFloat(obj.clip_similarity);
        return obj;
    });
}

// Utility: Fisher–Yates shuffle

function shuffle(array) {
    let a = array.slice();
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

// Load & preprocess CSV
fetch(CSV_FILE)
  .then(resp => resp.text())
  .then(text => {
    df = parseCSV(text).filter(r => !isNaN(r.clip_similarity));

    // Assign each row to a bin index
    df.forEach((row,i) => {
      for (let j = 0; j < CUSTOM_BINS.length - 1; j++) {
        let low  = CUSTOM_BINS[j],
            high = CUSTOM_BINS[j+1];
        if ((row.clip_similarity >= low && row.clip_similarity < high) ||
            (j===CUSTOM_BINS.length-2 && row.clip_similarity===high)) {
          row.bin = j;
          break;
        }
      }
    });

    // Collect & sort available bin indexes
    possibleValues = [...new Set(df.map(r=>r.bin))];
    possibleValues.sort((a,b)=>a-b);

    // Build per‐bin index lists, shuffle, cap at 10
   /* for (let b of possibleValues) {
      let allIdx = df
        .map((r,i)=> r.bin===b ? i : -1)
        .filter(i=>i!==-1);
      imgOrderPerBin[b] = shuffle(allIdx).slice(0,10);  // ← cap here
      usedIdxInBin[b]   = 0;
    }*/
    // Build per‐bin index lists and shuffle (no cap)
    for (let b of possibleValues) {
      let allIdx = df
        .map((r,i)=> r.bin===b ? i : -1)
        .filter(i=>i!==-1);
      imgOrderPerBin[b] = shuffle(allIdx);
      usedIdxInBin[b]   = 0;
    }

    // Create two staircases
    function makeStair(startHigh) {
      let startIdx = startHigh
        ? possibleValues.indexOf(Math.max(...possibleValues))
        : possibleValues.indexOf(Math.min(...possibleValues));
      return {
        currentIndex:    startIdx,
        currentBin:      possibleValues[startIdx],
        prevResponse:    null,
        reversalIndices: [],
        trialCount:      0,
        finished:        false
      };
    }
    staircases      = [ makeStair(false), makeStair(true) ];
    whichStairIndex = 0;
});

// init psychoJS:
const psychoJS = new PsychoJS({
  debug: true
});

// open window:
psychoJS.openWindow({
  fullscr: true,
  color: new util.Color([0,0,0]),
  units: 'height',
  waitBlanking: true,
  backgroundImage: '',
  backgroundFit: 'none',
});
// schedule the experiment:
psychoJS.schedule(psychoJS.gui.DlgFromDict({
  dictionary: expInfo,
  title: expName
}));

const flowScheduler = new Scheduler(psychoJS);
const dialogCancelScheduler = new Scheduler(psychoJS);
psychoJS.scheduleCondition(function() { return (psychoJS.gui.dialogComponent.button === 'OK'); },flowScheduler, dialogCancelScheduler);

// flowScheduler gets run if the participants presses OK
flowScheduler.add(updateInfo); // add timeStamp
flowScheduler.add(experimentInit);
flowScheduler.add(ConsentFormRoutineBegin());
flowScheduler.add(ConsentFormRoutineEachFrame());
flowScheduler.add(ConsentFormRoutineEnd());
flowScheduler.add(WelcomeRoutineBegin());
flowScheduler.add(WelcomeRoutineEachFrame());
flowScheduler.add(WelcomeRoutineEnd());
const trialsLoopScheduler = new Scheduler(psychoJS);
flowScheduler.add(trialsLoopBegin(trialsLoopScheduler));
flowScheduler.add(trialsLoopScheduler);
flowScheduler.add(trialsLoopEnd);


flowScheduler.add(QuestionaireRoutineBegin());
flowScheduler.add(QuestionaireRoutineEachFrame());
flowScheduler.add(QuestionaireRoutineEnd());
flowScheduler.add(thank_youRoutineBegin());
flowScheduler.add(thank_youRoutineEachFrame());
flowScheduler.add(thank_youRoutineEnd());
flowScheduler.add(quitPsychoJS, 'Thank you for your patience.', true);

// quit if user presses Cancel in dialog box:
dialogCancelScheduler.add(quitPsychoJS, 'Thank you for your patience.', false);

psychoJS.start({
  expName: expName,
  expInfo: expInfo,
  resources: [
    // resources:
    {'name': 'default.png', 'path': 'https://pavlovia.org/assets/default/default.png'},
    {'name': 'images/originals_needed/106.jpg', 'path': 'images/originals_needed/106.jpg'},
    {'name': 'images/originals_needed/150.jpg', 'path': 'images/originals_needed/150.jpg'},
    {'name': 'images/originals_needed/162.jpg', 'path': 'images/originals_needed/162.jpg'},
    {'name': 'images/originals_needed/845.jpg', 'path': 'images/originals_needed/845.jpg'},
    {'name': 'images/originals_needed/1837.jpg', 'path': 'images/originals_needed/1837.jpg'},
    {'name': 'images/originals_needed/2516.jpg', 'path': 'images/originals_needed/2516.jpg'},
    {'name': 'images/originals_needed/2589.jpg', 'path': 'images/originals_needed/2589.jpg'},
    {'name': 'images/originals_needed/2892.jpg', 'path': 'images/originals_needed/2892.jpg'},
    {'name': 'images/originals_needed/3463.jpg', 'path': 'images/originals_needed/3463.jpg'},
    {'name': 'images/originals_needed/3791.jpg', 'path': 'images/originals_needed/3791.jpg'},
    {'name': 'images/originals_needed/4374.jpg', 'path': 'images/originals_needed/4374.jpg'},
    {'name': 'images/originals_needed/4785.jpg', 'path': 'images/originals_needed/4785.jpg'},
    {'name': 'images/originals_needed/5139.jpg', 'path': 'images/originals_needed/5139.jpg'},
    {'name': 'images/originals_needed/6864.jpg', 'path': 'images/originals_needed/6864.jpg'},
    {'name': 'images/originals_needed/7416.jpg', 'path': 'images/originals_needed/7416.jpg'},
    {'name': 'images/originals_needed/7701.jpg', 'path': 'images/originals_needed/7701.jpg'},
    {'name': 'images/originals_needed/7867.jpg', 'path': 'images/originals_needed/7867.jpg'},
    {'name': 'images/originals_needed/8201.jpg', 'path': 'images/originals_needed/8201.jpg'},
    {'name': 'images/originals_needed/8709.jpg', 'path': 'images/originals_needed/8709.jpg'},
    {'name': 'images/originals_needed/8791.jpg', 'path': 'images/originals_needed/8791.jpg'},
    {'name': 'images/originals_needed/9190.jpg', 'path': 'images/originals_needed/9190.jpg'},
    {'name': 'images/originals_needed/9273.jpg', 'path': 'images/originals_needed/9273.jpg'},
    {'name': 'images/originals_needed/9425.jpg', 'path': 'images/originals_needed/9425.jpg'},
    {'name': 'images/originals_needed/9482.jpg', 'path': 'images/originals_needed/9482.jpg'},
    {'name': 'images/originals_needed/9548.jpg', 'path': 'images/originals_needed/9548.jpg'},
    {'name': 'images/originals_needed/9580.jpg', 'path': 'images/originals_needed/9580.jpg'},
    {'name': 'images/originals_needed/10053.jpg', 'path': 'images/originals_needed/10053.jpg'},
    {'name': 'images/originals_needed/10237.jpg', 'path': 'images/originals_needed/10237.jpg'},
    {'name': 'images/originals_needed/11120.jpg', 'path': 'images/originals_needed/11120.jpg'},
    {'name': 'images/originals_needed/12363.jpg', 'path': 'images/originals_needed/12363.jpg'},
    {'name': 'images/originals_needed/12966.jpg', 'path': 'images/originals_needed/12966.jpg'},
    {'name': 'images/originals_needed/13275.jpg', 'path': 'images/originals_needed/13275.jpg'},
    {'name': 'images/originals_needed/13471.jpg', 'path': 'images/originals_needed/13471.jpg'},
    {'name': 'images/originals_needed/13978.jpg', 'path': 'images/originals_needed/13978.jpg'},
    {'name': 'images/originals_needed/18921.jpg', 'path': 'images/originals_needed/18921.jpg'},
    {'name': 'images/originals_needed/20682.jpg', 'path': 'images/originals_needed/20682.jpg'},
    {'name': 'images/originals_needed/20686.jpg', 'path': 'images/originals_needed/20686.jpg'},
    {'name': 'images/originals_needed/20771.jpg', 'path': 'images/originals_needed/20771.jpg'},
    {'name': 'images/originals_needed/21347.jpg', 'path': 'images/originals_needed/21347.jpg'},
    {'name': 'images/originals_needed/21678.jpg', 'path': 'images/originals_needed/21678.jpg'},
    {'name': 'images/originals_needed/23375.jpg', 'path': 'images/originals_needed/23375.jpg'},
    {'name': 'images/originals_needed/23379.jpg', 'path': 'images/originals_needed/23379.jpg'},
    {'name': 'images/originals_needed/23684.jpg', 'path': 'images/originals_needed/23684.jpg'},
    {'name': 'images/originals_needed/24102.jpg', 'path': 'images/originals_needed/24102.jpg'},
    {'name': 'images/originals_needed/24313.jpg', 'path': 'images/originals_needed/24313.jpg'},
    {'name': 'images/originals_needed/24838.jpg', 'path': 'images/originals_needed/24838.jpg'},
    {'name': 'images/originals_needed/26634.jpg', 'path': 'images/originals_needed/26634.jpg'},
    {'name': 'images/originals_needed/31628.jpg', 'path': 'images/originals_needed/31628.jpg'},
    {'name': 'images/originals_needed/32163.jpg', 'path': 'images/originals_needed/32163.jpg'},
    {'name': 'images/originals_needed/33625.jpg', 'path': 'images/originals_needed/33625.jpg'},
    {'name': 'images/originals_needed/33761.jpg', 'path': 'images/originals_needed/33761.jpg'},
    {'name': 'images/originals_needed/35072.jpg', 'path': 'images/originals_needed/35072.jpg'},
    {'name': 'images/originals_needed/35182.jpg', 'path': 'images/originals_needed/35182.jpg'},
    {'name': 'images/originals_needed/35507.jpg', 'path': 'images/originals_needed/35507.jpg'},
    {'name': 'images/originals_needed/44289.jpg', 'path': 'images/originals_needed/44289.jpg'},
    {'name': 'images/originals_needed/44321.jpg', 'path': 'images/originals_needed/44321.jpg'},
    {'name': 'images/originals_needed/45563.jpg', 'path': 'images/originals_needed/45563.jpg'},
    {'name': 'images/originals_needed/46416.jpg', 'path': 'images/originals_needed/46416.jpg'},
    {'name': 'images/originals_needed/51974.jpg', 'path': 'images/originals_needed/51974.jpg'},
    {'name': 'images/originals_needed/52602.jpg', 'path': 'images/originals_needed/52602.jpg'},
    {'name': 'images/originals_needed/55587.jpg', 'path': 'images/originals_needed/55587.jpg'},
    {'name': 'images/originals_needed/55938.jpg', 'path': 'images/originals_needed/55938.jpg'},
    {'name': 'images/originals_needed/59214.jpg', 'path': 'images/originals_needed/59214.jpg'},
    {'name': 'images/originals_needed/60583.jpg', 'path': 'images/originals_needed/60583.jpg'},
    {'name': 'images/originals_needed/60588.jpg', 'path': 'images/originals_needed/60588.jpg'},
    {'name': 'images/originals_needed/60745.jpg', 'path': 'images/originals_needed/60745.jpg'},
    {'name': 'images/originals_needed/62065.jpg', 'path': 'images/originals_needed/62065.jpg'},
    {'name': 'images/originals_needed/62168.jpg', 'path': 'images/originals_needed/62168.jpg'},
    {'name': 'images/originals_needed/64527.jpg', 'path': 'images/originals_needed/64527.jpg'},
    {'name': 'images/originals_needed/70123.jpg', 'path': 'images/originals_needed/70123.jpg'},
    {'name': 'images/originals_needed/70160.jpg', 'path': 'images/originals_needed/70160.jpg'},
    {'name': 'images/originals_needed/73221.jpg', 'path': 'images/originals_needed/73221.jpg'},
    {'name': 'images/originals_needed/74340.jpg', 'path': 'images/originals_needed/74340.jpg'},
    {'name': 'images/originals_needed/74403.jpg', 'path': 'images/originals_needed/74403.jpg'},
    {'name': 'images/originals_needed/75572.jpg', 'path': 'images/originals_needed/75572.jpg'},
    {'name': 'images/originals_needed/80815.jpg', 'path': 'images/originals_needed/80815.jpg'},
    {'name': 'images/originals_needed/81685.jpg', 'path': 'images/originals_needed/81685.jpg'},
    {'name': 'images/originals_needed/82345.jpg', 'path': 'images/originals_needed/82345.jpg'},
    {'name': 'images/originals_needed/85569.jpg', 'path': 'images/originals_needed/85569.jpg'},
    {'name': 'images/originals_needed/86213.jpg', 'path': 'images/originals_needed/86213.jpg'},
    {'name': 'images/originals_needed/92663.jpg', 'path': 'images/originals_needed/92663.jpg'},
    {'name': 'images/originals_needed/93210.jpg', 'path': 'images/originals_needed/93210.jpg'},
    {'name': 'images/originals_needed/95217.jpg', 'path': 'images/originals_needed/95217.jpg'},
    {'name': 'images/originals_needed/95281.jpg', 'path': 'images/originals_needed/95281.jpg'},
    {'name': 'images/originals_needed/96800.jpg', 'path': 'images/originals_needed/96800.jpg'},
    {'name': 'images/originals_needed/98088.jpg', 'path': 'images/originals_needed/98088.jpg'},
    {'name': 'images/originals_needed/100397.jpg', 'path': 'images/originals_needed/100397.jpg'},
    {'name': 'images/originals_needed/104023.jpg', 'path': 'images/originals_needed/104023.jpg'},
    {'name': 'images/originals_needed/105190.jpg', 'path': 'images/originals_needed/105190.jpg'},
    {'name': 'images/originals_needed/106183.jpg', 'path': 'images/originals_needed/106183.jpg'},
    {'name': 'images/originals_needed/106707.jpg', 'path': 'images/originals_needed/106707.jpg'},
    {'name': 'images/originals_needed/111547.jpg', 'path': 'images/originals_needed/111547.jpg'},
    {'name': 'images/originals_needed/112993.jpg', 'path': 'images/originals_needed/112993.jpg'},
    {'name': 'images/originals_needed/118238.jpg', 'path': 'images/originals_needed/118238.jpg'},
    {'name': 'images/originals_needed/129910.jpg', 'path': 'images/originals_needed/129910.jpg'},
    {'name': 'images/originals_needed/142717.jpg', 'path': 'images/originals_needed/142717.jpg'},
    {'name': 'images/originals_needed/146420.jpg', 'path': 'images/originals_needed/146420.jpg'},
    {'name': 'images/originals_needed/151108.jpg', 'path': 'images/originals_needed/151108.jpg'},
    {'name': 'images/originals_needed/151628.jpg', 'path': 'images/originals_needed/151628.jpg'},
    {'name': 'images/originals_needed/152095.jpg', 'path': 'images/originals_needed/152095.jpg'},
    {'name': 'images/originals_needed/154133.jpg', 'path': 'images/originals_needed/154133.jpg'},
    {'name': 'images/originals_needed/158188.jpg', 'path': 'images/originals_needed/158188.jpg'},
    {'name': 'images/originals_needed/161996.jpg', 'path': 'images/originals_needed/161996.jpg'},
    {'name': 'images/originals_needed/162251.jpg', 'path': 'images/originals_needed/162251.jpg'},
    {'name': 'images/originals_needed/167612.jpg', 'path': 'images/originals_needed/167612.jpg'},
    {'name': 'images/originals_needed/170254.jpg', 'path': 'images/originals_needed/170254.jpg'},
    {'name': 'images/originals_needed/175666.jpg', 'path': 'images/originals_needed/175666.jpg'},
    {'name': 'images/originals_needed/181188.jpg', 'path': 'images/originals_needed/181188.jpg'},
    {'name': 'images/originals_needed/182112.jpg', 'path': 'images/originals_needed/182112.jpg'},
    {'name': 'images/originals_needed/182657.jpg', 'path': 'images/originals_needed/182657.jpg'},
    {'name': 'images/originals_needed/184714.jpg', 'path': 'images/originals_needed/184714.jpg'},
    {'name': 'images/originals_needed/187930.jpg', 'path': 'images/originals_needed/187930.jpg'},
    {'name': 'images/originals_needed/200460.jpg', 'path': 'images/originals_needed/200460.jpg'},
    {'name': 'images/originals_needed/202320.jpg', 'path': 'images/originals_needed/202320.jpg'},
    {'name': 'images/originals_needed/202870.jpg', 'path': 'images/originals_needed/202870.jpg'},
    {'name': 'images/originals_needed/205140.jpg', 'path': 'images/originals_needed/205140.jpg'},
    {'name': 'images/originals_needed/206282.jpg', 'path': 'images/originals_needed/206282.jpg'},
    {'name': 'images/originals_needed/210609.jpg', 'path': 'images/originals_needed/210609.jpg'},
    {'name': 'images/originals_needed/212523.jpg', 'path': 'images/originals_needed/212523.jpg'},
    {'name': 'images/originals_needed/213464.jpg', 'path': 'images/originals_needed/213464.jpg'},
    {'name': 'images/originals_needed/219683.jpg', 'path': 'images/originals_needed/219683.jpg'},
    {'name': 'images/originals_needed/224501.jpg', 'path': 'images/originals_needed/224501.jpg'},
    {'name': 'images/originals_needed/224554.jpg', 'path': 'images/originals_needed/224554.jpg'},
    {'name': 'images/originals_needed/230418.jpg', 'path': 'images/originals_needed/230418.jpg'},
    {'name': 'images/originals_needed/234535.jpg', 'path': 'images/originals_needed/234535.jpg'},
    {'name': 'images/originals_needed/236562.jpg', 'path': 'images/originals_needed/236562.jpg'},
    {'name': 'images/originals_needed/244379.jpg', 'path': 'images/originals_needed/244379.jpg'},
    {'name': 'images/originals_needed/245934.jpg', 'path': 'images/originals_needed/245934.jpg'},
    {'name': 'images/originals_needed/254035.jpg', 'path': 'images/originals_needed/254035.jpg'},
    {'name': 'images/originals_needed/257327.jpg', 'path': 'images/originals_needed/257327.jpg'},
    {'name': 'images/originals_needed/260639.jpg', 'path': 'images/originals_needed/260639.jpg'},
    {'name': 'images/originals_needed/262355.jpg', 'path': 'images/originals_needed/262355.jpg'},
    {'name': 'images/originals_needed/267110.jpg', 'path': 'images/originals_needed/267110.jpg'},
    {'name': 'images/originals_needed/269775.jpg', 'path': 'images/originals_needed/269775.jpg'},
    {'name': 'images/originals_needed/271059.jpg', 'path': 'images/originals_needed/271059.jpg'},
    {'name': 'images/originals_needed/271336.jpg', 'path': 'images/originals_needed/271336.jpg'},
    {'name': 'images/originals_needed/271488.jpg', 'path': 'images/originals_needed/271488.jpg'},
    {'name': 'images/originals_needed/281626.jpg', 'path': 'images/originals_needed/281626.jpg'},
    {'name': 'images/originals_needed/282357.jpg', 'path': 'images/originals_needed/282357.jpg'},
    {'name': 'images/originals_needed/284583.jpg', 'path': 'images/originals_needed/284583.jpg'},
    {'name': 'images/originals_needed/288838.jpg', 'path': 'images/originals_needed/288838.jpg'},
    {'name': 'images/originals_needed/297417.jpg', 'path': 'images/originals_needed/297417.jpg'},
    {'name': 'images/originals_needed/304135.jpg', 'path': 'images/originals_needed/304135.jpg'},
    {'name': 'images/originals_needed/310485.jpg', 'path': 'images/originals_needed/310485.jpg'},
    {'name': 'images/originals_needed/311346.jpg', 'path': 'images/originals_needed/311346.jpg'},
    {'name': 'images/originals_needed/315045.jpg', 'path': 'images/originals_needed/315045.jpg'},
    {'name': 'images/originals_needed/316215.jpg', 'path': 'images/originals_needed/316215.jpg'},
    {'name': 'images/originals_needed/316307.jpg', 'path': 'images/originals_needed/316307.jpg'},
    {'name': 'images/originals_needed/321020.jpg', 'path': 'images/originals_needed/321020.jpg'},
    {'name': 'images/originals_needed/322092.jpg', 'path': 'images/originals_needed/322092.jpg'},
    {'name': 'images/originals_needed/324088.jpg', 'path': 'images/originals_needed/324088.jpg'},
    {'name': 'images/originals_needed/328322.jpg', 'path': 'images/originals_needed/328322.jpg'},
    {'name': 'images/originals_needed/331127.jpg', 'path': 'images/originals_needed/331127.jpg'},
    {'name': 'images/originals_needed/332710.jpg', 'path': 'images/originals_needed/332710.jpg'},
    {'name': 'images/originals_needed/333630.jpg', 'path': 'images/originals_needed/333630.jpg'},
    {'name': 'images/originals_needed/335951.jpg', 'path': 'images/originals_needed/335951.jpg'},
    {'name': 'images/originals_needed/338938.jpg', 'path': 'images/originals_needed/338938.jpg'},
    {'name': 'images/originals_needed/340665.jpg', 'path': 'images/originals_needed/340665.jpg'},
    {'name': 'images/originals_needed/347044.jpg', 'path': 'images/originals_needed/347044.jpg'},
    {'name': 'images/originals_needed/354075.jpg', 'path': 'images/originals_needed/354075.jpg'},
    {'name': 'images/originals_needed/364795.jpg', 'path': 'images/originals_needed/364795.jpg'},
    {'name': 'images/originals_needed/365807.jpg', 'path': 'images/originals_needed/365807.jpg'},
    {'name': 'images/originals_needed/375809.jpg', 'path': 'images/originals_needed/375809.jpg'},
    {'name': 'images/originals_needed/378166.jpg', 'path': 'images/originals_needed/378166.jpg'},
    {'name': 'images/originals_needed/379860.jpg', 'path': 'images/originals_needed/379860.jpg'},
    {'name': 'images/originals_needed/382368.jpg', 'path': 'images/originals_needed/382368.jpg'},
    {'name': 'images/originals_needed/385155.jpg', 'path': 'images/originals_needed/385155.jpg'},
    {'name': 'images/originals_needed/386696.jpg', 'path': 'images/originals_needed/386696.jpg'},
    {'name': 'images/originals_needed/391314.jpg', 'path': 'images/originals_needed/391314.jpg'},
    {'name': 'images/originals_needed/391422.jpg', 'path': 'images/originals_needed/391422.jpg'},
    {'name': 'images/originals_needed/404406.jpg', 'path': 'images/originals_needed/404406.jpg'},
    {'name': 'images/originals_needed/404498.jpg', 'path': 'images/originals_needed/404498.jpg'},
    {'name': 'images/originals_needed/409812.jpg', 'path': 'images/originals_needed/409812.jpg'},
    {'name': 'images/originals_needed/412313.jpg', 'path': 'images/originals_needed/412313.jpg'},
    {'name': 'images/originals_needed/421771.jpg', 'path': 'images/originals_needed/421771.jpg'},
    {'name': 'images/originals_needed/424359.jpg', 'path': 'images/originals_needed/424359.jpg'},
    {'name': 'images/originals_needed/424449.jpg', 'path': 'images/originals_needed/424449.jpg'},
    {'name': 'images/originals_needed/432920.jpg', 'path': 'images/originals_needed/432920.jpg'},
    {'name': 'images/originals_needed/449235.jpg', 'path': 'images/originals_needed/449235.jpg'},
    {'name': 'images/originals_needed/452849.jpg', 'path': 'images/originals_needed/452849.jpg'},
    {'name': 'images/originals_needed/460138.jpg', 'path': 'images/originals_needed/460138.jpg'},
    {'name': 'images/originals_needed/468444.jpg', 'path': 'images/originals_needed/468444.jpg'},
    {'name': 'images/originals_needed/471810.jpg', 'path': 'images/originals_needed/471810.jpg'},
    {'name': 'images/originals_needed/477472.jpg', 'path': 'images/originals_needed/477472.jpg'},
    {'name': 'images/originals_needed/478531.jpg', 'path': 'images/originals_needed/478531.jpg'},
    {'name': 'images/originals_needed/478905.jpg', 'path': 'images/originals_needed/478905.jpg'},
    {'name': 'images/originals_needed/490999.jpg', 'path': 'images/originals_needed/490999.jpg'},
    {'name': 'images/originals_needed/491369.jpg', 'path': 'images/originals_needed/491369.jpg'},
    {'name': 'images/originals_needed/491860.jpg', 'path': 'images/originals_needed/491860.jpg'},
    {'name': 'images/originals_needed/493189.jpg', 'path': 'images/originals_needed/493189.jpg'},
    {'name': 'images/originals_needed/494261.jpg', 'path': 'images/originals_needed/494261.jpg'},
    {'name': 'images/originals_needed/501841.jpg', 'path': 'images/originals_needed/501841.jpg'},
    {'name': 'images/originals_needed/502691.jpg', 'path': 'images/originals_needed/502691.jpg'},
    {'name': 'images/originals_needed/506212.jpg', 'path': 'images/originals_needed/506212.jpg'},
    {'name': 'images/originals_needed/509164.jpg', 'path': 'images/originals_needed/509164.jpg'},
    {'name': 'images/originals_needed/509400.jpg', 'path': 'images/originals_needed/509400.jpg'},
    {'name': 'images/originals_needed/511442.jpg', 'path': 'images/originals_needed/511442.jpg'},
    {'name': 'images/originals_needed/516715.jpg', 'path': 'images/originals_needed/516715.jpg'},
    {'name': 'images/originals_needed/524086.jpg', 'path': 'images/originals_needed/524086.jpg'},
    {'name': 'images/originals_needed/526883.jpg', 'path': 'images/originals_needed/526883.jpg'},
    {'name': 'images/originals_needed/536211.jpg', 'path': 'images/originals_needed/536211.jpg'},
    {'name': 'images/originals_needed/538345.jpg', 'path': 'images/originals_needed/538345.jpg'},
    {'name': 'images/originals_needed/538845.jpg', 'path': 'images/originals_needed/538845.jpg'},
    {'name': 'images/originals_needed/543104.jpg', 'path': 'images/originals_needed/543104.jpg'},
    {'name': 'images/originals_needed/550619.jpg', 'path': 'images/originals_needed/550619.jpg'},
    {'name': 'images/originals_needed/569876.jpg', 'path': 'images/originals_needed/569876.jpg'},
    {'name': 'images/originals_needed/575469.jpg', 'path': 'images/originals_needed/575469.jpg'},
    {'name': 'images/originals_needed/575515.jpg', 'path': 'images/originals_needed/575515.jpg'},
    {'name': 'images/originals_needed/604740.jpg', 'path': 'images/originals_needed/604740.jpg'},
    {'name': 'images/originals_needed/619746.jpg', 'path': 'images/originals_needed/619746.jpg'},
    {'name': 'images/originals_needed/623459.jpg', 'path': 'images/originals_needed/623459.jpg'},
    {'name': 'images/originals_needed/623645.jpg', 'path': 'images/originals_needed/623645.jpg'},
    {'name': 'images/originals_needed/623703.jpg', 'path': 'images/originals_needed/623703.jpg'},
    {'name': 'images/originals_needed/624606.jpg', 'path': 'images/originals_needed/624606.jpg'},
    {'name': 'images/originals_needed/641334.jpg', 'path': 'images/originals_needed/641334.jpg'},
    {'name': 'images/originals_needed/642962.jpg', 'path': 'images/originals_needed/642962.jpg'},
    {'name': 'images/originals_needed/649485.jpg', 'path': 'images/originals_needed/649485.jpg'},
    {'name': 'images/originals_needed/650596.jpg', 'path': 'images/originals_needed/650596.jpg'},
    {'name': 'images/originals_needed/659202.jpg', 'path': 'images/originals_needed/659202.jpg'},
    {'name': 'images/originals_needed/669367.jpg', 'path': 'images/originals_needed/669367.jpg'},
    {'name': 'images/originals_needed/676523.jpg', 'path': 'images/originals_needed/676523.jpg'},
    {'name': 'images/originals_needed/679749.jpg', 'path': 'images/originals_needed/679749.jpg'},
    {'name': 'images/originals_needed/683341.jpg', 'path': 'images/originals_needed/683341.jpg'},
    {'name': 'images/originals_needed/685066.jpg', 'path': 'images/originals_needed/685066.jpg'},
    {'name': 'images/originals_needed/687274.jpg', 'path': 'images/originals_needed/687274.jpg'},
    {'name': 'images/originals_needed/693398.jpg', 'path': 'images/originals_needed/693398.jpg'},
    {'name': 'images/originals_needed/698076.jpg', 'path': 'images/originals_needed/698076.jpg'},
    {'name': 'images/originals_needed/704494.jpg', 'path': 'images/originals_needed/704494.jpg'},
    {'name': 'images/originals_needed/710917.jpg', 'path': 'images/originals_needed/710917.jpg'},
    {'name': 'images/originals_needed/713942.jpg', 'path': 'images/originals_needed/713942.jpg'},
    {'name': 'images/originals_needed/716381.jpg', 'path': 'images/originals_needed/716381.jpg'},
    {'name': 'images/originals_needed/721906.jpg', 'path': 'images/originals_needed/721906.jpg'},
    {'name': 'images/originals_needed/723297.jpg', 'path': 'images/originals_needed/723297.jpg'},
    {'name': 'images/originals_needed/723480.jpg', 'path': 'images/originals_needed/723480.jpg'},
    {'name': 'images/originals_needed/725129.jpg', 'path': 'images/originals_needed/725129.jpg'},
    {'name': 'images/originals_needed/727495.jpg', 'path': 'images/originals_needed/727495.jpg'},
    {'name': 'images/originals_needed/738537.jpg', 'path': 'images/originals_needed/738537.jpg'},
    {'name': 'images/originals_needed/739685.jpg', 'path': 'images/originals_needed/739685.jpg'},
    {'name': 'images/originals_needed/742675.jpg', 'path': 'images/originals_needed/742675.jpg'},
    {'name': 'images/originals_needed/745204.jpg', 'path': 'images/originals_needed/745204.jpg'},
    {'name': 'images/originals_needed/745635.jpg', 'path': 'images/originals_needed/745635.jpg'},
    {'name': 'images/originals_needed/747175.jpg', 'path': 'images/originals_needed/747175.jpg'},
    {'name': 'images/originals_needed/750117.jpg', 'path': 'images/originals_needed/750117.jpg'},
    {'name': 'images/originals_needed/758276.jpg', 'path': 'images/originals_needed/758276.jpg'},
    {'name': 'images/originals_needed/762843.jpg', 'path': 'images/originals_needed/762843.jpg'},
    {'name': 'images/originals_needed/764581.jpg', 'path': 'images/originals_needed/764581.jpg'},
    {'name': 'images/originals_needed/769605.jpg', 'path': 'images/originals_needed/769605.jpg'},
    {'name': 'images/originals_needed/770297.jpg', 'path': 'images/originals_needed/770297.jpg'},
    {'name': 'images/originals_needed/770432.jpg', 'path': 'images/originals_needed/770432.jpg'},
    {'name': 'images/originals_needed/771472.jpg', 'path': 'images/originals_needed/771472.jpg'},
    {'name': 'images/originals_needed/776681.jpg', 'path': 'images/originals_needed/776681.jpg'},
    {'name': 'images/originals_needed/780373.jpg', 'path': 'images/originals_needed/780373.jpg'},
    {'name': 'images/originals_needed/782420.jpg', 'path': 'images/originals_needed/782420.jpg'},
    {'name': 'images/originals_needed/793952.jpg', 'path': 'images/originals_needed/793952.jpg'},
    {'name': 'images/originals_needed/820923.jpg', 'path': 'images/originals_needed/820923.jpg'},
    {'name': 'images/originals_needed/821454.jpg', 'path': 'images/originals_needed/821454.jpg'},
    {'name': 'images/originals_needed/829060.jpg', 'path': 'images/originals_needed/829060.jpg'},
    {'name': 'images/originals_needed/830744.jpg', 'path': 'images/originals_needed/830744.jpg'},
    {'name': 'images/originals_needed/831802.jpg', 'path': 'images/originals_needed/831802.jpg'},
    {'name': 'images/originals_needed/834679.jpg', 'path': 'images/originals_needed/834679.jpg'},
    {'name': 'images/originals_needed/835590.jpg', 'path': 'images/originals_needed/835590.jpg'},
    {'name': 'images/originals_needed/835838.jpg', 'path': 'images/originals_needed/835838.jpg'},
    {'name': 'images/originals_needed/858475.jpg', 'path': 'images/originals_needed/858475.jpg'},
    {'name': 'images/originals_needed/861086.jpg', 'path': 'images/originals_needed/861086.jpg'},
    {'name': 'images/originals_needed/862470.jpg', 'path': 'images/originals_needed/862470.jpg'},
    {'name': 'images/originals_needed/867694.jpg', 'path': 'images/originals_needed/867694.jpg'},
    {'name': 'images/originals_needed/869672.jpg', 'path': 'images/originals_needed/869672.jpg'},
    {'name': 'images/originals_needed/874684.jpg', 'path': 'images/originals_needed/874684.jpg'},
    {'name': 'images/originals_needed/878987.jpg', 'path': 'images/originals_needed/878987.jpg'},
    {'name': 'images/originals_needed/885604.jpg', 'path': 'images/originals_needed/885604.jpg'},
    {'name': 'images/originals_needed/899376.jpg', 'path': 'images/originals_needed/899376.jpg'},
    {'name': 'images/originals_needed/899548.jpg', 'path': 'images/originals_needed/899548.jpg'},
    {'name': 'images/originals_needed/901323.jpg', 'path': 'images/originals_needed/901323.jpg'},
    {'name': 'images/originals_needed/903287.jpg', 'path': 'images/originals_needed/903287.jpg'},
    {'name': 'images/originals_needed/913046.jpg', 'path': 'images/originals_needed/913046.jpg'},
    {'name': 'images/originals_needed/918646.jpg', 'path': 'images/originals_needed/918646.jpg'},
    {'name': 'images/originals_needed/941726.jpg', 'path': 'images/originals_needed/941726.jpg'},
    {'name': 'images/originals_needed/947347.jpg', 'path': 'images/originals_needed/947347.jpg'},
    {'name': 'images/originals_needed/949475.jpg', 'path': 'images/originals_needed/949475.jpg'},
    {'name': 'images/originals_needed/952960.jpg', 'path': 'images/originals_needed/952960.jpg'},
    {'name': 'images/originals_needed/957023.jpg', 'path': 'images/originals_needed/957023.jpg'},
    {'name': 'images/originals_needed/957982.jpg', 'path': 'images/originals_needed/957982.jpg'},
    {'name': 'images/generated/106_generated.png', 'path': 'images/generated/106_generated.png'},
    {'name': 'images/generated/150_generated.png', 'path': 'images/generated/150_generated.png'},
    {'name': 'images/generated/162_generated.png', 'path': 'images/generated/162_generated.png'},
    {'name': 'images/generated/845_generated.png', 'path': 'images/generated/845_generated.png'},
    {'name': 'images/generated/1837_generated.png', 'path': 'images/generated/1837_generated.png'},
    {'name': 'images/generated/2516_generated.png', 'path': 'images/generated/2516_generated.png'},
    {'name': 'images/generated/2589_generated.png', 'path': 'images/generated/2589_generated.png'},
    {'name': 'images/generated/2892_generated.png', 'path': 'images/generated/2892_generated.png'},
    {'name': 'images/generated/3463_generated.png', 'path': 'images/generated/3463_generated.png'},
    {'name': 'images/generated/3791_generated.png', 'path': 'images/generated/3791_generated.png'},
    {'name': 'images/generated/4374_generated.png', 'path': 'images/generated/4374_generated.png'},
    {'name': 'images/generated/4785_generated.png', 'path': 'images/generated/4785_generated.png'},
    {'name': 'images/generated/5139_generated.png', 'path': 'images/generated/5139_generated.png'},
    {'name': 'images/generated/6864_generated.png', 'path': 'images/generated/6864_generated.png'},
    {'name': 'images/generated/7416_generated.png', 'path': 'images/generated/7416_generated.png'},
    {'name': 'images/generated/7701_generated.png', 'path': 'images/generated/7701_generated.png'},
    {'name': 'images/generated/7867_generated.png', 'path': 'images/generated/7867_generated.png'},
    {'name': 'images/generated/8201_generated.png', 'path': 'images/generated/8201_generated.png'},
    {'name': 'images/generated/8709_generated.png', 'path': 'images/generated/8709_generated.png'},
    {'name': 'images/generated/8791_generated.png', 'path': 'images/generated/8791_generated.png'},
    {'name': 'images/generated/9190_generated.png', 'path': 'images/generated/9190_generated.png'},
    {'name': 'images/generated/9273_generated.png', 'path': 'images/generated/9273_generated.png'},
    {'name': 'images/generated/9425_generated.png', 'path': 'images/generated/9425_generated.png'},
    {'name': 'images/generated/9482_generated.png', 'path': 'images/generated/9482_generated.png'},
    {'name': 'images/generated/9548_generated.png', 'path': 'images/generated/9548_generated.png'},
    {'name': 'images/generated/9580_generated.png', 'path': 'images/generated/9580_generated.png'},
    {'name': 'images/generated/10053_generated.png', 'path': 'images/generated/10053_generated.png'},
    {'name': 'images/generated/10237_generated.png', 'path': 'images/generated/10237_generated.png'},
    {'name': 'images/generated/11120_generated.png', 'path': 'images/generated/11120_generated.png'},
    {'name': 'images/generated/12363_generated.png', 'path': 'images/generated/12363_generated.png'},
    {'name': 'images/generated/12966_generated.png', 'path': 'images/generated/12966_generated.png'},
    {'name': 'images/generated/13275_generated.png', 'path': 'images/generated/13275_generated.png'},
    {'name': 'images/generated/13471_generated.png', 'path': 'images/generated/13471_generated.png'},
    {'name': 'images/generated/13978_generated.png', 'path': 'images/generated/13978_generated.png'},
    {'name': 'images/generated/18921_generated.png', 'path': 'images/generated/18921_generated.png'},
    {'name': 'images/generated/20682_generated.png', 'path': 'images/generated/20682_generated.png'},
    {'name': 'images/generated/20686_generated.png', 'path': 'images/generated/20686_generated.png'},
    {'name': 'images/generated/20771_generated.png', 'path': 'images/generated/20771_generated.png'},
    {'name': 'images/generated/21347_generated.png', 'path': 'images/generated/21347_generated.png'},
    {'name': 'images/generated/21678_generated.png', 'path': 'images/generated/21678_generated.png'},
    {'name': 'images/generated/23375_generated.png', 'path': 'images/generated/23375_generated.png'},
    {'name': 'images/generated/23379_generated.png', 'path': 'images/generated/23379_generated.png'},
    {'name': 'images/generated/23684_generated.png', 'path': 'images/generated/23684_generated.png'},
    {'name': 'images/generated/24102_generated.png', 'path': 'images/generated/24102_generated.png'},
    {'name': 'images/generated/24313_generated.png', 'path': 'images/generated/24313_generated.png'},
    {'name': 'images/generated/24838_generated.png', 'path': 'images/generated/24838_generated.png'},
    {'name': 'images/generated/26634_generated.png', 'path': 'images/generated/26634_generated.png'},
    {'name': 'images/generated/31628_generated.png', 'path': 'images/generated/31628_generated.png'},
    {'name': 'images/generated/32163_generated.png', 'path': 'images/generated/32163_generated.png'},
    {'name': 'images/generated/33625_generated.png', 'path': 'images/generated/33625_generated.png'},
    {'name': 'images/generated/33761_generated.png', 'path': 'images/generated/33761_generated.png'},
    {'name': 'images/generated/35072_generated.png', 'path': 'images/generated/35072_generated.png'},
    {'name': 'images/generated/35182_generated.png', 'path': 'images/generated/35182_generated.png'},
    {'name': 'images/generated/35507_generated.png', 'path': 'images/generated/35507_generated.png'},
    {'name': 'images/generated/44289_generated.png', 'path': 'images/generated/44289_generated.png'},
    {'name': 'images/generated/44321_generated.png', 'path': 'images/generated/44321_generated.png'},
    {'name': 'images/generated/45563_generated.png', 'path': 'images/generated/45563_generated.png'},
    {'name': 'images/generated/46416_generated.png', 'path': 'images/generated/46416_generated.png'},
    {'name': 'images/generated/51974_generated.png', 'path': 'images/generated/51974_generated.png'},
    {'name': 'images/generated/52602_generated.png', 'path': 'images/generated/52602_generated.png'},
    {'name': 'images/generated/55587_generated.png', 'path': 'images/generated/55587_generated.png'},
    {'name': 'images/generated/55938_generated.png', 'path': 'images/generated/55938_generated.png'},
    {'name': 'images/generated/59214_generated.png', 'path': 'images/generated/59214_generated.png'},
    {'name': 'images/generated/60583_generated.png', 'path': 'images/generated/60583_generated.png'},
    {'name': 'images/generated/60588_generated.png', 'path': 'images/generated/60588_generated.png'},
    {'name': 'images/generated/60745_generated.png', 'path': 'images/generated/60745_generated.png'},
    {'name': 'images/generated/62065_generated.png', 'path': 'images/generated/62065_generated.png'},
    {'name': 'images/generated/62168_generated.png', 'path': 'images/generated/62168_generated.png'},
    {'name': 'images/generated/64527_generated.png', 'path': 'images/generated/64527_generated.png'},
    {'name': 'images/generated/70123_generated.png', 'path': 'images/generated/70123_generated.png'},
    {'name': 'images/generated/70160_generated.png', 'path': 'images/generated/70160_generated.png'},
    {'name': 'images/generated/73221_generated.png', 'path': 'images/generated/73221_generated.png'},
    {'name': 'images/generated/74340_generated.png', 'path': 'images/generated/74340_generated.png'},
    {'name': 'images/generated/74403_generated.png', 'path': 'images/generated/74403_generated.png'},
    {'name': 'images/generated/75572_generated.png', 'path': 'images/generated/75572_generated.png'},
    {'name': 'images/generated/80815_generated.png', 'path': 'images/generated/80815_generated.png'},
    {'name': 'images/generated/81685_generated.png', 'path': 'images/generated/81685_generated.png'},
    {'name': 'images/generated/82345_generated.png', 'path': 'images/generated/82345_generated.png'},
    {'name': 'images/generated/85569_generated.png', 'path': 'images/generated/85569_generated.png'},
    {'name': 'images/generated/86213_generated.png', 'path': 'images/generated/86213_generated.png'},
    {'name': 'images/generated/92663_generated.png', 'path': 'images/generated/92663_generated.png'},
    {'name': 'images/generated/93210_generated.png', 'path': 'images/generated/93210_generated.png'},
    {'name': 'images/generated/95217_generated.png', 'path': 'images/generated/95217_generated.png'},
    {'name': 'images/generated/95281_generated.png', 'path': 'images/generated/95281_generated.png'},
    {'name': 'images/generated/96800_generated.png', 'path': 'images/generated/96800_generated.png'},
    {'name': 'images/generated/98088_generated.png', 'path': 'images/generated/98088_generated.png'},
    {'name': 'images/generated/100397_generated.png', 'path': 'images/generated/100397_generated.png'},
    {'name': 'images/generated/104023_generated.png', 'path': 'images/generated/104023_generated.png'},
    {'name': 'images/generated/105190_generated.png', 'path': 'images/generated/105190_generated.png'},
    {'name': 'images/generated/106183_generated.png', 'path': 'images/generated/106183_generated.png'},
    {'name': 'images/generated/106707_generated.png', 'path': 'images/generated/106707_generated.png'},
    {'name': 'images/generated/111547_generated.png', 'path': 'images/generated/111547_generated.png'},
    {'name': 'images/generated/112993_generated.png', 'path': 'images/generated/112993_generated.png'},
    {'name': 'images/generated/118238_generated.png', 'path': 'images/generated/118238_generated.png'},
    {'name': 'images/generated/129910_generated.png', 'path': 'images/generated/129910_generated.png'},
    {'name': 'images/generated/142717_generated.png', 'path': 'images/generated/142717_generated.png'},
    {'name': 'images/generated/146420_generated.png', 'path': 'images/generated/146420_generated.png'},
    {'name': 'images/generated/151108_generated.png', 'path': 'images/generated/151108_generated.png'},
    {'name': 'images/generated/151628_generated.png', 'path': 'images/generated/151628_generated.png'},
    {'name': 'images/generated/152095_generated.png', 'path': 'images/generated/152095_generated.png'},
    {'name': 'images/generated/154133_generated.png', 'path': 'images/generated/154133_generated.png'},
    {'name': 'images/generated/158188_generated.png', 'path': 'images/generated/158188_generated.png'},
    {'name': 'images/generated/161996_generated.png', 'path': 'images/generated/161996_generated.png'},
    {'name': 'images/generated/162251_generated.png', 'path': 'images/generated/162251_generated.png'},
    {'name': 'images/generated/167612_generated.png', 'path': 'images/generated/167612_generated.png'},
    {'name': 'images/generated/170254_generated.png', 'path': 'images/generated/170254_generated.png'},
    {'name': 'images/generated/175666_generated.png', 'path': 'images/generated/175666_generated.png'},
    {'name': 'images/generated/181188_generated.png', 'path': 'images/generated/181188_generated.png'},
    {'name': 'images/generated/182112_generated.png', 'path': 'images/generated/182112_generated.png'},
    {'name': 'images/generated/182657_generated.png', 'path': 'images/generated/182657_generated.png'},
    {'name': 'images/generated/184714_generated.png', 'path': 'images/generated/184714_generated.png'},
    {'name': 'images/generated/187930_generated.png', 'path': 'images/generated/187930_generated.png'},
    {'name': 'images/generated/200460_generated.png', 'path': 'images/generated/200460_generated.png'},
    {'name': 'images/generated/202320_generated.png', 'path': 'images/generated/202320_generated.png'},
    {'name': 'images/generated/202870_generated.png', 'path': 'images/generated/202870_generated.png'},
    {'name': 'images/generated/205140_generated.png', 'path': 'images/generated/205140_generated.png'},
    {'name': 'images/generated/206282_generated.png', 'path': 'images/generated/206282_generated.png'},
    {'name': 'images/generated/210609_generated.png', 'path': 'images/generated/210609_generated.png'},
    {'name': 'images/generated/212523_generated.png', 'path': 'images/generated/212523_generated.png'},
    {'name': 'images/generated/213464_generated.png', 'path': 'images/generated/213464_generated.png'},
    {'name': 'images/generated/219683_generated.png', 'path': 'images/generated/219683_generated.png'},
    {'name': 'images/generated/224501_generated.png', 'path': 'images/generated/224501_generated.png'},
    {'name': 'images/generated/224554_generated.png', 'path': 'images/generated/224554_generated.png'},
    {'name': 'images/generated/230418_generated.png', 'path': 'images/generated/230418_generated.png'},
    {'name': 'images/generated/234535_generated.png', 'path': 'images/generated/234535_generated.png'},
    {'name': 'images/generated/236562_generated.png', 'path': 'images/generated/236562_generated.png'},
    {'name': 'images/generated/244379_generated.png', 'path': 'images/generated/244379_generated.png'},
    {'name': 'images/generated/245934_generated.png', 'path': 'images/generated/245934_generated.png'},
    {'name': 'images/generated/254035_generated.png', 'path': 'images/generated/254035_generated.png'},
    {'name': 'images/generated/257327_generated.png', 'path': 'images/generated/257327_generated.png'},
    {'name': 'images/generated/260639_generated.png', 'path': 'images/generated/260639_generated.png'},
    {'name': 'images/generated/262355_generated.png', 'path': 'images/generated/262355_generated.png'},
    {'name': 'images/generated/267110_generated.png', 'path': 'images/generated/267110_generated.png'},
    {'name': 'images/generated/269775_generated.png', 'path': 'images/generated/269775_generated.png'},
    {'name': 'images/generated/271059_generated.png', 'path': 'images/generated/271059_generated.png'},
    {'name': 'images/generated/271336_generated.png', 'path': 'images/generated/271336_generated.png'},
    {'name': 'images/generated/271488_generated.png', 'path': 'images/generated/271488_generated.png'},
    {'name': 'images/generated/281626_generated.png', 'path': 'images/generated/281626_generated.png'},
    {'name': 'images/generated/282357_generated.png', 'path': 'images/generated/282357_generated.png'},
    {'name': 'images/generated/284583_generated.png', 'path': 'images/generated/284583_generated.png'},
    {'name': 'images/generated/288838_generated.png', 'path': 'images/generated/288838_generated.png'},
    {'name': 'images/generated/297417_generated.png', 'path': 'images/generated/297417_generated.png'},
    {'name': 'images/generated/304135_generated.png', 'path': 'images/generated/304135_generated.png'},
    {'name': 'images/generated/310485_generated.png', 'path': 'images/generated/310485_generated.png'},
    {'name': 'images/generated/311346_generated.png', 'path': 'images/generated/311346_generated.png'},
    {'name': 'images/generated/315045_generated.png', 'path': 'images/generated/315045_generated.png'},
    {'name': 'images/generated/316215_generated.png', 'path': 'images/generated/316215_generated.png'},
    {'name': 'images/generated/316307_generated.png', 'path': 'images/generated/316307_generated.png'},
    {'name': 'images/generated/321020_generated.png', 'path': 'images/generated/321020_generated.png'},
    {'name': 'images/generated/322092_generated.png', 'path': 'images/generated/322092_generated.png'},
    {'name': 'images/generated/324088_generated.png', 'path': 'images/generated/324088_generated.png'},
    {'name': 'images/generated/328322_generated.png', 'path': 'images/generated/328322_generated.png'},
    {'name': 'images/generated/331127_generated.png', 'path': 'images/generated/331127_generated.png'},
    {'name': 'images/generated/332710_generated.png', 'path': 'images/generated/332710_generated.png'},
    {'name': 'images/generated/333630_generated.png', 'path': 'images/generated/333630_generated.png'},
    {'name': 'images/generated/335951_generated.png', 'path': 'images/generated/335951_generated.png'},
    {'name': 'images/generated/338938_generated.png', 'path': 'images/generated/338938_generated.png'},
    {'name': 'images/generated/340665_generated.png', 'path': 'images/generated/340665_generated.png'},
    {'name': 'images/generated/347044_generated.png', 'path': 'images/generated/347044_generated.png'},
    {'name': 'images/generated/354075_generated.png', 'path': 'images/generated/354075_generated.png'},
    {'name': 'images/generated/364795_generated.png', 'path': 'images/generated/364795_generated.png'},
    {'name': 'images/generated/365807_generated.png', 'path': 'images/generated/365807_generated.png'},
    {'name': 'images/generated/375809_generated.png', 'path': 'images/generated/375809_generated.png'},
    {'name': 'images/generated/378166_generated.png', 'path': 'images/generated/378166_generated.png'},
    {'name': 'images/generated/379860_generated.png', 'path': 'images/generated/379860_generated.png'},
    {'name': 'images/generated/382368_generated.png', 'path': 'images/generated/382368_generated.png'},
    {'name': 'images/generated/385155_generated.png', 'path': 'images/generated/385155_generated.png'},
    {'name': 'images/generated/386696_generated.png', 'path': 'images/generated/386696_generated.png'},
    {'name': 'images/generated/391314_generated.png', 'path': 'images/generated/391314_generated.png'},
    {'name': 'images/generated/391422_generated.png', 'path': 'images/generated/391422_generated.png'},
    {'name': 'images/generated/404406_generated.png', 'path': 'images/generated/404406_generated.png'},
    {'name': 'images/generated/404498_generated.png', 'path': 'images/generated/404498_generated.png'},
    {'name': 'images/generated/409812_generated.png', 'path': 'images/generated/409812_generated.png'},
    {'name': 'images/generated/412313_generated.png', 'path': 'images/generated/412313_generated.png'},
    {'name': 'images/generated/421771_generated.png', 'path': 'images/generated/421771_generated.png'},
    {'name': 'images/generated/424359_generated.png', 'path': 'images/generated/424359_generated.png'},
    {'name': 'images/generated/424449_generated.png', 'path': 'images/generated/424449_generated.png'},
    {'name': 'images/generated/432920_generated.png', 'path': 'images/generated/432920_generated.png'},
    {'name': 'images/generated/449235_generated.png', 'path': 'images/generated/449235_generated.png'},
    {'name': 'images/generated/452849_generated.png', 'path': 'images/generated/452849_generated.png'},
    {'name': 'images/generated/460138_generated.png', 'path': 'images/generated/460138_generated.png'},
    {'name': 'images/generated/468444_generated.png', 'path': 'images/generated/468444_generated.png'},
    {'name': 'images/generated/471810_generated.png', 'path': 'images/generated/471810_generated.png'},
    {'name': 'images/generated/477472_generated.png', 'path': 'images/generated/477472_generated.png'},
    {'name': 'images/generated/478531_generated.png', 'path': 'images/generated/478531_generated.png'},
    {'name': 'images/generated/478905_generated.png', 'path': 'images/generated/478905_generated.png'},
    {'name': 'images/generated/490999_generated.png', 'path': 'images/generated/490999_generated.png'},
    {'name': 'images/generated/491369_generated.png', 'path': 'images/generated/491369_generated.png'},
    {'name': 'images/generated/491860_generated.png', 'path': 'images/generated/491860_generated.png'},
    {'name': 'images/generated/493189_generated.png', 'path': 'images/generated/493189_generated.png'},
    {'name': 'images/generated/494261_generated.png', 'path': 'images/generated/494261_generated.png'},
    {'name': 'images/generated/501841_generated.png', 'path': 'images/generated/501841_generated.png'},
    {'name': 'images/generated/502691_generated.png', 'path': 'images/generated/502691_generated.png'},
    {'name': 'images/generated/506212_generated.png', 'path': 'images/generated/506212_generated.png'},
    {'name': 'images/generated/509164_generated.png', 'path': 'images/generated/509164_generated.png'},
    {'name': 'images/generated/509400_generated.png', 'path': 'images/generated/509400_generated.png'},
    {'name': 'images/generated/511442_generated.png', 'path': 'images/generated/511442_generated.png'},
    {'name': 'images/generated/516715_generated.png', 'path': 'images/generated/516715_generated.png'},
    {'name': 'images/generated/524086_generated.png', 'path': 'images/generated/524086_generated.png'},
    {'name': 'images/generated/526883_generated.png', 'path': 'images/generated/526883_generated.png'},
    {'name': 'images/generated/536211_generated.png', 'path': 'images/generated/536211_generated.png'},
    {'name': 'images/generated/538345_generated.png', 'path': 'images/generated/538345_generated.png'},
    {'name': 'images/generated/538845_generated.png', 'path': 'images/generated/538845_generated.png'},
    {'name': 'images/generated/543104_generated.png', 'path': 'images/generated/543104_generated.png'},
    {'name': 'images/generated/550619_generated.png', 'path': 'images/generated/550619_generated.png'},
    {'name': 'images/generated/569876_generated.png', 'path': 'images/generated/569876_generated.png'},
    {'name': 'images/generated/575469_generated.png', 'path': 'images/generated/575469_generated.png'},
    {'name': 'images/generated/575515_generated.png', 'path': 'images/generated/575515_generated.png'},
    {'name': 'images/generated/604740_generated.png', 'path': 'images/generated/604740_generated.png'},
    {'name': 'images/generated/619746_generated.png', 'path': 'images/generated/619746_generated.png'},
    {'name': 'images/generated/623459_generated.png', 'path': 'images/generated/623459_generated.png'},
    {'name': 'images/generated/623645_generated.png', 'path': 'images/generated/623645_generated.png'},
    {'name': 'images/generated/623703_generated.png', 'path': 'images/generated/623703_generated.png'},
    {'name': 'images/generated/624606_generated.png', 'path': 'images/generated/624606_generated.png'},
    {'name': 'images/generated/641334_generated.png', 'path': 'images/generated/641334_generated.png'},
    {'name': 'images/generated/642962_generated.png', 'path': 'images/generated/642962_generated.png'},
    {'name': 'images/generated/649485_generated.png', 'path': 'images/generated/649485_generated.png'},
    {'name': 'images/generated/650596_generated.png', 'path': 'images/generated/650596_generated.png'},
    {'name': 'images/generated/659202_generated.png', 'path': 'images/generated/659202_generated.png'},
    {'name': 'images/generated/669367_generated.png', 'path': 'images/generated/669367_generated.png'},
    {'name': 'images/generated/676523_generated.png', 'path': 'images/generated/676523_generated.png'},
    {'name': 'images/generated/679749_generated.png', 'path': 'images/generated/679749_generated.png'},
    {'name': 'images/generated/683341_generated.png', 'path': 'images/generated/683341_generated.png'},
    {'name': 'images/generated/685066_generated.png', 'path': 'images/generated/685066_generated.png'},
    {'name': 'images/generated/687274_generated.png', 'path': 'images/generated/687274_generated.png'},
    {'name': 'images/generated/693398_generated.png', 'path': 'images/generated/693398_generated.png'},
    {'name': 'images/generated/698076_generated.png', 'path': 'images/generated/698076_generated.png'},
    {'name': 'images/generated/704494_generated.png', 'path': 'images/generated/704494_generated.png'},
    {'name': 'images/generated/710917_generated.png', 'path': 'images/generated/710917_generated.png'},
    {'name': 'images/generated/713942_generated.png', 'path': 'images/generated/713942_generated.png'},
    {'name': 'images/generated/716381_generated.png', 'path': 'images/generated/716381_generated.png'},
    {'name': 'images/generated/721906_generated.png', 'path': 'images/generated/721906_generated.png'},
    {'name': 'images/generated/723297_generated.png', 'path': 'images/generated/723297_generated.png'},
    {'name': 'images/generated/723480_generated.png', 'path': 'images/generated/723480_generated.png'},
    {'name': 'images/generated/725129_generated.png', 'path': 'images/generated/725129_generated.png'},
    {'name': 'images/generated/727495_generated.png', 'path': 'images/generated/727495_generated.png'},
    {'name': 'images/generated/738537_generated.png', 'path': 'images/generated/738537_generated.png'},
    {'name': 'images/generated/739685_generated.png', 'path': 'images/generated/739685_generated.png'},
    {'name': 'images/generated/742675_generated.png', 'path': 'images/generated/742675_generated.png'},
    {'name': 'images/generated/745204_generated.png', 'path': 'images/generated/745204_generated.png'},
    {'name': 'images/generated/745635_generated.png', 'path': 'images/generated/745635_generated.png'},
    {'name': 'images/generated/747175_generated.png', 'path': 'images/generated/747175_generated.png'},
    {'name': 'images/generated/750117_generated.png', 'path': 'images/generated/750117_generated.png'},
    {'name': 'images/generated/758276_generated.png', 'path': 'images/generated/758276_generated.png'},
    {'name': 'images/generated/762843_generated.png', 'path': 'images/generated/762843_generated.png'},
    {'name': 'images/generated/764581_generated.png', 'path': 'images/generated/764581_generated.png'},
    {'name': 'images/generated/769605_generated.png', 'path': 'images/generated/769605_generated.png'},
    {'name': 'images/generated/770297_generated.png', 'path': 'images/generated/770297_generated.png'},
    {'name': 'images/generated/770432_generated.png', 'path': 'images/generated/770432_generated.png'},
    {'name': 'images/generated/771472_generated.png', 'path': 'images/generated/771472_generated.png'},
    {'name': 'images/generated/776681_generated.png', 'path': 'images/generated/776681_generated.png'},
    {'name': 'images/generated/780373_generated.png', 'path': 'images/generated/780373_generated.png'},
    {'name': 'images/generated/782420_generated.png', 'path': 'images/generated/782420_generated.png'},
    {'name': 'images/generated/793952_generated.png', 'path': 'images/generated/793952_generated.png'},
    {'name': 'images/generated/820923_generated.png', 'path': 'images/generated/820923_generated.png'},
    {'name': 'images/generated/821454_generated.png', 'path': 'images/generated/821454_generated.png'},
    {'name': 'images/generated/829060_generated.png', 'path': 'images/generated/829060_generated.png'},
    {'name': 'images/generated/830744_generated.png', 'path': 'images/generated/830744_generated.png'},
    {'name': 'images/generated/831802_generated.png', 'path': 'images/generated/831802_generated.png'},
    {'name': 'images/generated/834679_generated.png', 'path': 'images/generated/834679_generated.png'},
    {'name': 'images/generated/835590_generated.png', 'path': 'images/generated/835590_generated.png'},
    {'name': 'images/generated/835838_generated.png', 'path': 'images/generated/835838_generated.png'},
    {'name': 'images/generated/858475_generated.png', 'path': 'images/generated/858475_generated.png'},
    {'name': 'images/generated/861086_generated.png', 'path': 'images/generated/861086_generated.png'},
    {'name': 'images/generated/862470_generated.png', 'path': 'images/generated/862470_generated.png'},
    {'name': 'images/generated/867694_generated.png', 'path': 'images/generated/867694_generated.png'},
    {'name': 'images/generated/869672_generated.png', 'path': 'images/generated/869672_generated.png'},
    {'name': 'images/generated/874684_generated.png', 'path': 'images/generated/874684_generated.png'},
    {'name': 'images/generated/878987_generated.png', 'path': 'images/generated/878987_generated.png'},
    {'name': 'images/generated/885604_generated.png', 'path': 'images/generated/885604_generated.png'},
    {'name': 'images/generated/899376_generated.png', 'path': 'images/generated/899376_generated.png'},
    {'name': 'images/generated/899548_generated.png', 'path': 'images/generated/899548_generated.png'},
    {'name': 'images/generated/901323_generated.png', 'path': 'images/generated/901323_generated.png'},
    {'name': 'images/generated/903287_generated.png', 'path': 'images/generated/903287_generated.png'},
    {'name': 'images/generated/913046_generated.png', 'path': 'images/generated/913046_generated.png'},
    {'name': 'images/generated/918646_generated.png', 'path': 'images/generated/918646_generated.png'},
    {'name': 'images/generated/941726_generated.png', 'path': 'images/generated/941726_generated.png'},
    {'name': 'images/generated/947347_generated.png', 'path': 'images/generated/947347_generated.png'},
    {'name': 'images/generated/949475_generated.png', 'path': 'images/generated/949475_generated.png'},
    {'name': 'images/generated/952960_generated.png', 'path': 'images/generated/952960_generated.png'},
    {'name': 'images/generated/957023_generated.png', 'path': 'images/generated/957023_generated.png'},
    {'name': 'images/generated/957982_generated.png', 'path': 'images/generated/957982_generated.png'},
    {'name': 'csv/stimuli.csv', 'path': 'csv/stimuli.csv'},
  ]
});

psychoJS.experimentLogger.setLevel(core.Logger.ServerLevel.INFO);


var currentLoop;
var frameDur;
async function updateInfo() {
  currentLoop = psychoJS.experiment;  // right now there are no loops
  expInfo['date'] = util.MonotonicClock.getDateStr();  // add a simple timestamp
  expInfo['expName'] = expName;
  expInfo['psychopyVersion'] = '2024.2.4';
  expInfo['OS'] = window.navigator.platform;


  // store frame rate of monitor if we can measure it successfully
  expInfo['frameRate'] = psychoJS.window.getActualFrameRate();
  if (typeof expInfo['frameRate'] !== 'undefined')
    frameDur = 1.0 / Math.round(expInfo['frameRate']);
  else
    frameDur = 1.0 / 60.0; // couldn't get a reliable measure so guess

  // add info from the URL:
  util.addInfoFromUrl(expInfo);
  

  
  psychoJS.experiment.dataFileName = (("." + "/") + `data/${expInfo["participant"]}_${expName}_${expInfo["date"]}`);
  psychoJS.experiment.field_separator = '\t';


  return Scheduler.Event.NEXT;
}


var ConsentFormClock;
var text_5;
var WelcomeClock;
var text;
var key_resp_welcome;
var mouse_continue;
var trialClock;
var image1;
var image2;
var trial_info_str;
var mouse_resp;
var User_guide;
var key_resp;
var btn_notsim;
var btn_sim;
var QuestionaireClock;
var dummy;
var thank_youClock;
var text_3;
var globalClock;
var routineTimer;
async function experimentInit() {
  // Initialize components for Routine "ConsentForm"
  ConsentFormClock = new util.Clock();
  text_5 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_5',
    text: '',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: 0.0,
    depth: -1.0 
  });
  
  // Initialize components for Routine "Welcome"
  WelcomeClock = new util.Clock();
  text = new visual.TextStim({
    win: psychoJS.window,
    name: 'text',
    text: 'Welcome!\n\nIn each trial you’ll see two images side by side. Your task is to decide whether they’re Similar or Not Similar.\n\nMouse: click the “Similar” or “Not Similar” button\n\nKeyboard: press → for Similar, ← for Not Similar\n\nWhen you’re ready, press SPACE (desktop) or tap anywhere on the screen (mobile) to begin.',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 0.04,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  key_resp_welcome = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  mouse_continue = new core.Mouse({
    win: psychoJS.window,
  });
  mouse_continue.mouseClock = new util.Clock();
  // Initialize components for Routine "trial"
  trialClock = new util.Clock();
  image1 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'image1', units : 'height', 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [(- 0.3), 0], 
    draggable: false,
    size : [0.5, 0.5],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -1.0 
  });
  image2 = new visual.ImageStim({
    win : psychoJS.window,
    name : 'image2', units : 'height', 
    image : 'default.png', mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0.3, 0], 
    draggable: false,
    size : [0.5, 0.5],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -2.0 
  });
  trial_info_str = new visual.TextStim({
    win: psychoJS.window,
    name: 'trial_info_str',
    text: '',
    font: 'Arial',
    units: undefined, 
    pos: [0, (+ 0.3)], draggable: false, height: 0.03,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -3.0 
  });
  
  mouse_resp = new core.Mouse({
    win: psychoJS.window,
  });
  mouse_resp.mouseClock = new util.Clock();
  User_guide = new visual.TextStim({
    win: psychoJS.window,
    name: 'User_guide',
    text: 'From Keybaord: Not Similar (←), Similar (→)',
    font: 'Arial',
    units: undefined, 
    pos: [0, (- 0.3)], draggable: false, height: 0.02,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: -5.0 
  });
  
  key_resp = new core.Keyboard({psychoJS: psychoJS, clock: new util.Clock(), waitForStart: true});
  
  btn_notsim = new visual.ButtonStim({
    win: psychoJS.window,
    name: 'btn_notsim',
    text: 'Not Similar',
    fillColor: 'darkgrey',
    borderColor: null,
    color: 'white',
    colorSpace: 'rgb',
    pos: [(- 0.3), (- 0.4)],
    letterHeight: 0.03,
    size: [0.3, 0.08],
    ori: 0.0
    ,
    depth: -7
  });
  btn_notsim.clock = new util.Clock();
  
  btn_sim = new visual.ButtonStim({
    win: psychoJS.window,
    name: 'btn_sim',
    text: 'Similar',
    fillColor: 'darkgrey',
    borderColor: null,
    color: 'white',
    colorSpace: 'rgb',
    pos: [(+ 0.3), (- 0.4)],
    letterHeight: 0.03,
    size: [0.3, 0.08],
    ori: 0.0
    ,
    depth: -8
  });
  btn_sim.clock = new util.Clock();
  
  // Initialize components for Routine "Questionaire"
  QuestionaireClock = new util.Clock();
  // =============================================================================
  // BEGIN EXPERIMENT TAB
  // =============================================================================
  // Initialize global variables for questionnaire
  window.rankingComplete = false;
  window.selectedItems = [];
  window.rankingContainer = null;
  dummy = new visual.ImageStim({
    win : psychoJS.window,
    name : 'dummy', units : undefined, 
    image : undefined, mask : undefined,
    anchor : 'center',
    ori : 0.0, 
    pos : [0, 0], 
    draggable: false,
    size : [0.5, 0.5],
    color : new util.Color([1,1,1]), opacity : undefined,
    flipHoriz : false, flipVert : false,
    texRes : 128.0, interpolate : true, depth : -1.0 
  });
  // Initialize components for Routine "thank_you"
  thank_youClock = new util.Clock();
  text_3 = new visual.TextStim({
    win: psychoJS.window,
    name: 'text_3',
    text: 'Thank you',
    font: 'Arial',
    units: undefined, 
    pos: [0, 0], draggable: false, height: 0.05,  wrapWidth: undefined, ori: 0.0,
    languageStyle: 'LTR',
    color: new util.Color('white'),  opacity: undefined,
    depth: 0.0 
  });
  
  // Create some handy timers
  globalClock = new util.Clock();  // to track the time since experiment started
  routineTimer = new util.CountdownTimer();  // to track time remaining of each (non-slip) routine
  
  return Scheduler.Event.NEXT;
}


var t;
var frameN;
var continueRoutine;
var ConsentFormMaxDurationReached;
var ConsentFormMaxDuration;
var ConsentFormComponents;
function ConsentFormRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'ConsentForm' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    ConsentFormClock.reset();
    routineTimer.reset();
    ConsentFormMaxDurationReached = false;
    // update component parameters for each repeat
    // Run 'Begin Routine' code from ConsentCode
    // Set flag to block continuation until consent is given
    window.userConsented = false;
    continueRoutine = true;
    
    // Create container
    const consentBox = document.createElement('div');
    consentBox.id = 'consentBox';
    consentBox.style.cssText = `
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      width: 90%;
      max-width: 800px;
      background: #fff;
      border: 2px solid #333;
      border-radius: 10px;
      padding: 30px;
      box-shadow: 0 4px 15px rgba(0,0,0,0.2);
      font-family: Arial, sans-serif;
      z-index: 1000;
      overflow-y: auto;
      max-height: 80vh;
    `;
    
    // Consent text
    const consentText = document.createElement('p');
    consentText.innerHTML = `
      <strong>Consent Form</strong><br><br>
    
    You are invited to participate in a research study on how people perceive visual similarity between images. This study is part of an academic project investigating human aesthetic preferences and visual cognition.<br><br>
    
    Participation involves viewing image pairs and making similarity judgments, followed by a short set of demographic questions (including age, gender, country of origin, and any background in photography or visual arts).<br><br>
    
    Your participation is entirely voluntary, and you may withdraw at any time without penalty. No personally identifiable information (such as your name or contact details) will be collected. All responses will be stored securely and analyzed only in anonymized form.<br><br>
    
    This research complies with the General Data Protection Regulation (GDPR) and institutional data privacy policies. Data will be used exclusively for academic research and not shared beyond the research team.<br><br>
    
    By checking the box below, you confirm that:
    <ul>
      <li>You are at least 18 years old</li>
      <li>You understand the purpose and procedures of this study</li>
      <li>You consent to participate voluntarily</li>
    </ul>
    
    Thank you for your contribution to this research. Click "I agree" if you consent.
    `;
    consentText.style.cssText = `
      font-size: 16px;
      margin-bottom: 20px;
    `;
    
    // Checkbox
    const consentCheckbox = document.createElement('input');
    consentCheckbox.type = 'checkbox';
    consentCheckbox.id = 'consentCheckbox';
    consentCheckbox.style.marginRight = '10px';
    
    const checkboxLabel = document.createElement('label');
    checkboxLabel.textContent = 'I agree to participate in this study.';
    checkboxLabel.htmlFor = 'consentCheckbox';
    
    // Continue button (initially disabled)
    const continueBtn = document.createElement('button');
    continueBtn.textContent = 'Continue';
    continueBtn.disabled = true;  // ← DISABLED by default
    continueBtn.style.cssText = `
      display: block;
      margin-top: 20px;
      padding: 10px 20px;
      background: #4CAF50;
      color: white;
      border: none;
      border-radius: 6px;
      font-size: 16px;
      cursor: not-allowed;
      opacity: 0.5;
    `;
    
    // Enable button when checkbox is checked
    consentCheckbox.addEventListener('change', function () {
      if (consentCheckbox.checked) {
        continueBtn.disabled = false;
        continueBtn.style.cursor = 'pointer';
        continueBtn.style.opacity = '1';
      } else {
        continueBtn.disabled = true;
        continueBtn.style.cursor = 'not-allowed';
        continueBtn.style.opacity = '0.5';
      }
    });
    
    // Continue action
    continueBtn.onclick = function () {
      window.userConsented = true;
      const el = document.getElementById('consentBox');
      if (el && el.parentNode) el.parentNode.removeChild(el);
      document.body.style.overflow = 'auto';
    };
    
    // Assemble UI
    consentBox.appendChild(consentText);
    consentBox.appendChild(consentCheckbox);
    consentBox.appendChild(checkboxLabel);
    consentBox.appendChild(continueBtn);
    document.body.appendChild(consentBox);
    document.body.style.overflow = 'hidden';
    
    psychoJS.experiment.addData('ConsentForm.started', globalClock.getTime());
    ConsentFormMaxDuration = null
    // keep track of which components have finished
    ConsentFormComponents = [];
    ConsentFormComponents.push(text_5);
    
    for (const thisComponent of ConsentFormComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function ConsentFormRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'ConsentForm' ---
    // get current time
    t = ConsentFormClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // Run 'Each Frame' code from ConsentCode
    // Prevent continuation until user consents
    if (window.userConsented) {
      continueRoutine = false;
    }
    
    
    // *text_5* updates
    if (t >= 0.0 && text_5.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_5.tStart = t;  // (not accounting for frame time here)
      text_5.frameNStart = frameN;  // exact frame index
      
      text_5.setAutoDraw(true);
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of ConsentFormComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function ConsentFormRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'ConsentForm' ---
    for (const thisComponent of ConsentFormComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('ConsentForm.stopped', globalClock.getTime());
    // Run 'End Routine' code from ConsentCode
    const el = document.getElementById('consentBox');
    if (el && el.parentNode) el.parentNode.removeChild(el);
    document.body.style.overflow = 'auto';
    
    // the Routine "ConsentForm" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var WelcomeMaxDurationReached;
var _key_resp_welcome_allKeys;
var gotValidClick;
var WelcomeMaxDuration;
var WelcomeComponents;
function WelcomeRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'Welcome' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    WelcomeClock.reset();
    routineTimer.reset();
    WelcomeMaxDurationReached = false;
    // update component parameters for each repeat
    key_resp_welcome.keys = undefined;
    key_resp_welcome.rt = undefined;
    _key_resp_welcome_allKeys = [];
    // setup some python lists for storing info about the mouse_continue
    // current position of the mouse:
    mouse_continue.x = [];
    mouse_continue.y = [];
    mouse_continue.leftButton = [];
    mouse_continue.midButton = [];
    mouse_continue.rightButton = [];
    mouse_continue.time = [];
    gotValidClick = false; // until a click is received
    psychoJS.experiment.addData('Welcome.started', globalClock.getTime());
    WelcomeMaxDuration = null
    // keep track of which components have finished
    WelcomeComponents = [];
    WelcomeComponents.push(text);
    WelcomeComponents.push(key_resp_welcome);
    WelcomeComponents.push(mouse_continue);
    
    for (const thisComponent of WelcomeComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


var prevButtonState;
var _mouseButtons;
var _mouseXYs;
function WelcomeRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'Welcome' ---
    // get current time
    t = WelcomeClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text* updates
    if (t >= 0.0 && text.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text.tStart = t;  // (not accounting for frame time here)
      text.frameNStart = frameN;  // exact frame index
      
      text.setAutoDraw(true);
    }
    
    
    // *key_resp_welcome* updates
    if (t >= 0.0 && key_resp_welcome.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp_welcome.tStart = t;  // (not accounting for frame time here)
      key_resp_welcome.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp_welcome.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp_welcome.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp_welcome.clearEvents(); });
    }
    
    if (key_resp_welcome.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp_welcome.getKeys({keyList: ['space'], waitRelease: false});
      _key_resp_welcome_allKeys = _key_resp_welcome_allKeys.concat(theseKeys);
      if (_key_resp_welcome_allKeys.length > 0) {
        key_resp_welcome.keys = _key_resp_welcome_allKeys[_key_resp_welcome_allKeys.length - 1].name;  // just the last key pressed
        key_resp_welcome.rt = _key_resp_welcome_allKeys[_key_resp_welcome_allKeys.length - 1].rt;
        key_resp_welcome.duration = _key_resp_welcome_allKeys[_key_resp_welcome_allKeys.length - 1].duration;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    // *mouse_continue* updates
    if (t >= 0.0 && mouse_continue.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      mouse_continue.tStart = t;  // (not accounting for frame time here)
      mouse_continue.frameNStart = frameN;  // exact frame index
      
      mouse_continue.status = PsychoJS.Status.STARTED;
      mouse_continue.mouseClock.reset();
      prevButtonState = mouse_continue.getPressed();  // if button is down already this ISN'T a new click
      }
    if (mouse_continue.status === PsychoJS.Status.STARTED) {  // only update if started and not finished!
      _mouseButtons = mouse_continue.getPressed();
      if (!_mouseButtons.every( (e,i,) => (e == prevButtonState[i]) )) { // button state changed?
        prevButtonState = _mouseButtons;
        if (_mouseButtons.reduce( (e, acc) => (e+acc) ) > 0) { // state changed to a new click
          _mouseXYs = mouse_continue.getPos();
          mouse_continue.x.push(_mouseXYs[0]);
          mouse_continue.y.push(_mouseXYs[1]);
          mouse_continue.leftButton.push(_mouseButtons[0]);
          mouse_continue.midButton.push(_mouseButtons[1]);
          mouse_continue.rightButton.push(_mouseButtons[2]);
          mouse_continue.time.push(mouse_continue.mouseClock.getTime());
          // end routine on response
          continueRoutine = false;
        }
      }
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of WelcomeComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function WelcomeRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'Welcome' ---
    for (const thisComponent of WelcomeComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('Welcome.stopped', globalClock.getTime());
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_resp_welcome.corr, level);
    }
    psychoJS.experiment.addData('key_resp_welcome.keys', key_resp_welcome.keys);
    if (typeof key_resp_welcome.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp_welcome.rt', key_resp_welcome.rt);
        psychoJS.experiment.addData('key_resp_welcome.duration', key_resp_welcome.duration);
        routineTimer.reset();
        }
    
    key_resp_welcome.stop();
    // store data for psychoJS.experiment (ExperimentHandler)
    psychoJS.experiment.addData('mouse_continue.x', mouse_continue.x);
    psychoJS.experiment.addData('mouse_continue.y', mouse_continue.y);
    psychoJS.experiment.addData('mouse_continue.leftButton', mouse_continue.leftButton);
    psychoJS.experiment.addData('mouse_continue.midButton', mouse_continue.midButton);
    psychoJS.experiment.addData('mouse_continue.rightButton', mouse_continue.rightButton);
    psychoJS.experiment.addData('mouse_continue.time', mouse_continue.time);
    
    // the Routine "Welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var trials;
function trialsLoopBegin(trialsLoopScheduler, snapshot) {
  return async function() {
    TrialHandler.fromSnapshot(snapshot); // update internal variables (.thisN etc) of the loop
    
    // set up handler to look after randomisation of conditions etc
    trials = new TrialHandler({
      psychoJS: psychoJS,
      nReps: 190, method: TrialHandler.Method.RANDOM,
      extraInfo: expInfo, originPath: undefined,
      trialList: undefined,
      seed: undefined, name: 'trials'
    });
    psychoJS.experiment.addLoop(trials); // add the loop to the experiment
    currentLoop = trials;  // we're now the current loop
    
    // Schedule all the trials in the trialList:
    for (const thisTrial of trials) {
      snapshot = trials.getSnapshot();
      trialsLoopScheduler.add(importConditions(snapshot));
      trialsLoopScheduler.add(trialRoutineBegin(snapshot));
      trialsLoopScheduler.add(trialRoutineEachFrame());
      trialsLoopScheduler.add(trialRoutineEnd(snapshot));
      trialsLoopScheduler.add(trialsLoopEndIteration(trialsLoopScheduler, snapshot));
    }
    
    return Scheduler.Event.NEXT;
  }
}


async function trialsLoopEnd() {
  // terminate loop
  psychoJS.experiment.removeLoop(trials);
  // update the current loop from the ExperimentHandler
  if (psychoJS.experiment._unfinishedLoops.length>0)
    currentLoop = psychoJS.experiment._unfinishedLoops.at(-1);
  else
    currentLoop = psychoJS.experiment;  // so we use addData from the experiment
  return Scheduler.Event.NEXT;
}


function trialsLoopEndIteration(scheduler, snapshot) {
  // ------Prepare for next entry------
  return async function () {
    if (typeof snapshot !== 'undefined') {
      // ------Check if user ended loop early------
      if (snapshot.finished) {
        // Check for and save orphaned data
        if (psychoJS.experiment.isEntryEmpty()) {
          psychoJS.experiment.nextEntry(snapshot);
        }
        scheduler.stop();
      }
    return Scheduler.Event.NEXT;
    }
  };
}


var trialMaxDurationReached;
var image1Resized;
var currentIndex;
var currentBin;
var currentRow;
var _key_resp_allKeys;
var trialMaxDuration;
var trialComponents;
function trialRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'trial' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    trialClock.reset();
    routineTimer.reset();
    trialMaxDurationReached = false;
    // update component parameters for each repeat
    // Run 'Begin Routine' code from code
    ///////////////////////
    // "Begin Routine" TAB
    ///////////////////////
    
    // Reset resize flag
    image1Resized = false;
    
    // Select the active staircase
    var sc = staircases[whichStairIndex];
    currentIndex = sc.currentIndex;   // for legacy code below
    currentBin   = sc.currentBin;
    
    // Build list of candidate bins (staircase‐style expanding)
    let candidateBins = [ currentBin ];
    let max_offset    = Math.max(...possibleValues.map(b => Math.abs(b - currentBin)));
    for (let offset = 1; offset <= max_offset; offset++) {
      let down = currentBin - offset,
          up   = currentBin + offset;
      if (possibleValues.includes(down)) candidateBins.push(down);
      if (possibleValues.includes(up))   candidateBins.push(up);
    }
    
    // Find a new unused image index from these bins
    let found = false, img_idx;
    for (let b of candidateBins) {
      let idxs = imgOrderPerBin[b] || [];
      while (usedIdxInBin[b] < idxs.length) {
        img_idx = idxs[ usedIdxInBin[b]++ ];
        if (!usedGlobalIndices.has(img_idx)) {
          usedGlobalIndices.add(img_idx);
          found = true;
          break;
        }
      }
      if (found) break;
    }
    
    // If no image found, mark this staircase finished
    if (!found || img_idx === undefined || !df[img_idx]) {
      sc.finished = true;
    
      // If both staircases are done, end the loop
      if (staircases.every(s => s.finished)) {
        trials.finished = true;
      }
      continueRoutine = false;
      currentRow      = null;
    } 
    else {
      // Otherwise set up the trial normally
      currentRow = df[img_idx];
      shownRows.push(currentRow);
    
      let origPath = ORIGINAL_FOLDER + currentRow.original_image;
      let genPath  = GENERATED_FOLDER  + currentRow.generated_image;
      let origLeft = Math.random() < 0.5;
    
      image1_path = origLeft ? origPath : genPath;
      image2_path = origLeft ? genPath  : origPath;
    
      //info_str = `Bin: ${BIN_LABELS[currentRow.bin]} | Similarity: ${currentRow.clip_similarity.toFixed(3)} | Trial ${sc.trialCount+1}`;
    }
    
    image1.setImage(image1_path);
    image2.setImage(image2_path);
    trial_info_str.setText(info_str);
    // setup some python lists for storing info about the mouse_resp
    // current position of the mouse:
    mouse_resp.x = [];
    mouse_resp.y = [];
    mouse_resp.leftButton = [];
    mouse_resp.midButton = [];
    mouse_resp.rightButton = [];
    mouse_resp.time = [];
    mouse_resp.clicked_name = [];
    gotValidClick = false; // until a click is received
    key_resp.keys = undefined;
    key_resp.rt = undefined;
    _key_resp_allKeys = [];
    // reset btn_notsim to account for continued clicks & clear times on/off
    btn_notsim.reset()
    // reset btn_sim to account for continued clicks & clear times on/off
    btn_sim.reset()
    psychoJS.experiment.addData('trial.started', globalClock.getTime());
    trialMaxDuration = null
    // keep track of which components have finished
    trialComponents = [];
    trialComponents.push(image1);
    trialComponents.push(image2);
    trialComponents.push(trial_info_str);
    trialComponents.push(mouse_resp);
    trialComponents.push(User_guide);
    trialComponents.push(key_resp);
    trialComponents.push(btn_notsim);
    trialComponents.push(btn_sim);
    
    for (const thisComponent of trialComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function trialRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'trial' ---
    // get current time
    t = trialClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    ////////////////////
    // "Each Frame" TAB
    ////////////////////
    
    // Helper to scale images
    function scaleToFitBox(imgComponent, targetSize=[0.5,0.5]) {
      if (!imgComponent._image ||
          !imgComponent._image.complete ||
          imgComponent._image.naturalWidth === 0) {
        return;
      }
      let iw = imgComponent._image.naturalWidth,
          ih = imgComponent._image.naturalHeight,
          boxW = targetSize[0],
          boxH = targetSize[1],
          imageAspect = iw/ih,
          boxAspect   = boxW/boxH;
    
      if (imageAspect > boxAspect) {
        imgComponent.size = [ boxW, boxW/imageAspect ];
      } else {
        imgComponent.size = [ boxH*imageAspect, boxH ];
      }
    }
    
    if (!image1Resized && image1._image && image2._image) {
      scaleToFitBox(image1, [0.5,0.5]);
      scaleToFitBox(image2, [0.5,0.5]);
      image1Resized = true;
    }
    
    
    // *image1* updates
    if (t >= 0.0 && image1.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      image1.tStart = t;  // (not accounting for frame time here)
      image1.frameNStart = frameN;  // exact frame index
      
      image1.setAutoDraw(true);
    }
    
    
    // *image2* updates
    if (t >= 0.0 && image2.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      image2.tStart = t;  // (not accounting for frame time here)
      image2.frameNStart = frameN;  // exact frame index
      
      image2.setAutoDraw(true);
    }
    
    
    // *trial_info_str* updates
    if (t >= 0.0 && trial_info_str.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      trial_info_str.tStart = t;  // (not accounting for frame time here)
      trial_info_str.frameNStart = frameN;  // exact frame index
      
      trial_info_str.setAutoDraw(true);
    }
    
    // *mouse_resp* updates
    if (t >= 0.0 && mouse_resp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      mouse_resp.tStart = t;  // (not accounting for frame time here)
      mouse_resp.frameNStart = frameN;  // exact frame index
      
      mouse_resp.status = PsychoJS.Status.STARTED;
      mouse_resp.mouseClock.reset();
      prevButtonState = mouse_resp.getPressed();  // if button is down already this ISN'T a new click
      }
    if (mouse_resp.status === PsychoJS.Status.STARTED) {  // only update if started and not finished!
      _mouseButtons = mouse_resp.getPressed();
      if (!_mouseButtons.every( (e,i,) => (e == prevButtonState[i]) )) { // button state changed?
        prevButtonState = _mouseButtons;
        if (_mouseButtons.reduce( (e, acc) => (e+acc) ) > 0) { // state changed to a new click
          // check if the mouse was inside our 'clickable' objects
          gotValidClick = false;
          mouse_resp.clickableObjects = eval([btn_sim, btn_notsim])
          ;// make sure the mouse's clickable objects are an array
          if (!Array.isArray(mouse_resp.clickableObjects)) {
              mouse_resp.clickableObjects = [mouse_resp.clickableObjects];
          }
          // iterate through clickable objects and check each
          for (const obj of mouse_resp.clickableObjects) {
              if (obj.contains(mouse_resp)) {
                  gotValidClick = true;
                  mouse_resp.clicked_name.push(obj.name);
              }
          }
          if (!gotValidClick) {
              mouse_resp.clicked_name.push(null);
          }
          _mouseXYs = mouse_resp.getPos();
          mouse_resp.x.push(_mouseXYs[0]);
          mouse_resp.y.push(_mouseXYs[1]);
          mouse_resp.leftButton.push(_mouseButtons[0]);
          mouse_resp.midButton.push(_mouseButtons[1]);
          mouse_resp.rightButton.push(_mouseButtons[2]);
          mouse_resp.time.push(mouse_resp.mouseClock.getTime());
          if (gotValidClick === true) { // end routine on response
            continueRoutine = false;
          }
        }
      }
    }
    
    // *User_guide* updates
    if (t >= 0.0 && User_guide.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      User_guide.tStart = t;  // (not accounting for frame time here)
      User_guide.frameNStart = frameN;  // exact frame index
      
      User_guide.setAutoDraw(true);
    }
    
    
    // *key_resp* updates
    if (t >= 0.0 && key_resp.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      key_resp.tStart = t;  // (not accounting for frame time here)
      key_resp.frameNStart = frameN;  // exact frame index
      
      // keyboard checking is just starting
      psychoJS.window.callOnFlip(function() { key_resp.clock.reset(); });  // t=0 on next screen flip
      psychoJS.window.callOnFlip(function() { key_resp.start(); }); // start on screen flip
      psychoJS.window.callOnFlip(function() { key_resp.clearEvents(); });
    }
    
    if (key_resp.status === PsychoJS.Status.STARTED) {
      let theseKeys = key_resp.getKeys({keyList: ['a', 'd', 'left', 'right'], waitRelease: false});
      _key_resp_allKeys = _key_resp_allKeys.concat(theseKeys);
      if (_key_resp_allKeys.length > 0) {
        key_resp.keys = _key_resp_allKeys[_key_resp_allKeys.length - 1].name;  // just the last key pressed
        key_resp.rt = _key_resp_allKeys[_key_resp_allKeys.length - 1].rt;
        key_resp.duration = _key_resp_allKeys[_key_resp_allKeys.length - 1].duration;
        // a response ends the routine
        continueRoutine = false;
      }
    }
    
    
    // *btn_notsim* updates
    if (t >= 0 && btn_notsim.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      btn_notsim.tStart = t;  // (not accounting for frame time here)
      btn_notsim.frameNStart = frameN;  // exact frame index
      
      btn_notsim.setAutoDraw(true);
    }
    
    if (btn_notsim.status === PsychoJS.Status.STARTED) {
      // check whether btn_notsim has been pressed
      if (btn_notsim.isClicked) {
        if (!btn_notsim.wasClicked) {
          // store time of first click
          btn_notsim.timesOn.push(btn_notsim.clock.getTime());
          // store time clicked until
          btn_notsim.timesOff.push(btn_notsim.clock.getTime());
        } else {
          // update time clicked until;
          btn_notsim.timesOff[btn_notsim.timesOff.length - 1] = btn_notsim.clock.getTime();
        }
        if (!btn_notsim.wasClicked) {
          // end routine when btn_notsim is clicked
          continueRoutine = false;
          
        }
        // if btn_notsim is still clicked next frame, it is not a new click
        btn_notsim.wasClicked = true;
      } else {
        // if btn_notsim is clicked next frame, it is a new click
        btn_notsim.wasClicked = false;
      }
    } else {
      // keep clock at 0 if btn_notsim hasn't started / has finished
      btn_notsim.clock.reset();
      // if btn_notsim is clicked next frame, it is a new click
      btn_notsim.wasClicked = false;
    }
    
    // *btn_sim* updates
    if (t >= 0 && btn_sim.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      btn_sim.tStart = t;  // (not accounting for frame time here)
      btn_sim.frameNStart = frameN;  // exact frame index
      
      btn_sim.setAutoDraw(true);
    }
    
    if (btn_sim.status === PsychoJS.Status.STARTED) {
      // check whether btn_sim has been pressed
      if (btn_sim.isClicked) {
        if (!btn_sim.wasClicked) {
          // store time of first click
          btn_sim.timesOn.push(btn_sim.clock.getTime());
          // store time clicked until
          btn_sim.timesOff.push(btn_sim.clock.getTime());
        } else {
          // update time clicked until;
          btn_sim.timesOff[btn_sim.timesOff.length - 1] = btn_sim.clock.getTime();
        }
        if (!btn_sim.wasClicked) {
          // end routine when btn_sim is clicked
          continueRoutine = false;
          
        }
        // if btn_sim is still clicked next frame, it is not a new click
        btn_sim.wasClicked = true;
      } else {
        // if btn_sim is clicked next frame, it is a new click
        btn_sim.wasClicked = false;
      }
    } else {
      // keep clock at 0 if btn_sim hasn't started / has finished
      btn_sim.clock.reset();
      // if btn_sim is clicked next frame, it is a new click
      btn_sim.wasClicked = false;
    }
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of trialComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


var resp;
var kp;
var rt;
var whichStairIndex;
function trialRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'trial' ---
    for (const thisComponent of trialComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('trial.stopped', globalClock.getTime());
    // Run 'End Routine' code from code
    ////////////////////
    // "End Routine" TAB
    ////////////////////
    
    let resp = null, kp='none', rt='';
    
    // Gather keyboard or mouse response as before
    if (key_resp.keys) {
      let kl = Array.isArray(key_resp.keys) ? key_resp.keys : [key_resp.keys];
      if (kl.includes('right') || kl.includes('d')) {
        resp = 1; kp = kl.slice(-1)[0];
      } else if (kl.includes('left') || kl.includes('a')) {
        resp = 0; kp = kl.slice(-1)[0];
      }
      rt = Array.isArray(key_resp.rt) ? (key_resp.rt.slice(-1)[0] || '') : key_resp.rt;
    }
    else if (mouse_resp.clicked_name) {
      if (mouse_resp.clicked_name.includes('btn_sim'))     { resp = 1; kp = 'mouse_sim'; }
      else if (mouse_resp.clicked_name.includes('btn_notsim')) { resp = 0; kp = 'mouse_notsim'; }
      rt = (mouse_resp.times && mouse_resp.times.length)
           ? mouse_resp.times.slice(-1)[0]
           : '';
    }
    
    if (currentRow) {
      // Log global history
      history.push(currentRow.bin);
      responses.push(resp);
      reaction_times.push(rt);
      key_presses.push(kp);
      orig_lefts.push(image1_path.includes(currentRow.original_image));
    
      // Update the active staircase
      var sc = staircases[whichStairIndex];
      sc.trialCount++;
    
      // reversal detection
      if (sc.prevResponse !== null && sc.prevResponse !== resp) {
        sc.reversalIndices.push(sc.trialCount - 1);
      }
      sc.prevResponse = resp;
    
      // staircase step
      if (resp === 0) {
        sc.currentIndex = Math.min(sc.currentIndex + STEPSIZE, possibleValues.length - 1);
      } else {
        sc.currentIndex = Math.max(sc.currentIndex - STEPSIZE, 0);
      }
      sc.currentBin = possibleValues[ sc.currentIndex ];
    
      // finish criterion
      /*if (sc.reversalIndices.length >= NUM_REVERSALS &&
          sc.trialCount       >= MIN_TRIALS) {
        sc.finished = true;
      }*/
      if (sc.reversalIndices.length >= NUM_REVERSALS) {
        sc.finished = true;
      }
    
      // alternate staircases
      //whichStairIndex = 1 - whichStairIndex;
      // RECORD which staircase this was
      stairIndices.push(whichStairIndex);
      // now alternate to the other staircase
      whichStairIndex = 1 - whichStairIndex;
    
      /*// if both done, end loop
      if (staircases.every(s => s.finished)) {
        trials.finished = true;
      }*/
        // now check overall stop: both staircases reversed enough AND total trials ≥ MIN_TRIALS
      const bothReversed = staircases.every(s => s.reversalIndices.length >= NUM_REVERSALS);
      const totalTrials  = staircases.reduce((sum, s) => sum + s.trialCount, 0);
      if (bothReversed && totalTrials >= MIN_TRIALS) {
        trials.finished = true;
      }
      
    }
    
    // store data for psychoJS.experiment (ExperimentHandler)
    psychoJS.experiment.addData('mouse_resp.x', mouse_resp.x);
    psychoJS.experiment.addData('mouse_resp.y', mouse_resp.y);
    psychoJS.experiment.addData('mouse_resp.leftButton', mouse_resp.leftButton);
    psychoJS.experiment.addData('mouse_resp.midButton', mouse_resp.midButton);
    psychoJS.experiment.addData('mouse_resp.rightButton', mouse_resp.rightButton);
    psychoJS.experiment.addData('mouse_resp.time', mouse_resp.time);
    psychoJS.experiment.addData('mouse_resp.clicked_name', mouse_resp.clicked_name);
    
    // update the trial handler
    if (currentLoop instanceof MultiStairHandler) {
      currentLoop.addResponse(key_resp.corr, level);
    }
    psychoJS.experiment.addData('key_resp.keys', key_resp.keys);
    if (typeof key_resp.keys !== 'undefined') {  // we had a response
        psychoJS.experiment.addData('key_resp.rt', key_resp.rt);
        psychoJS.experiment.addData('key_resp.duration', key_resp.duration);
        routineTimer.reset();
        }
    
    key_resp.stop();
    psychoJS.experiment.addData('btn_notsim.numClicks', btn_notsim.numClicks);
    psychoJS.experiment.addData('btn_notsim.timesOn', btn_notsim.timesOn);
    psychoJS.experiment.addData('btn_notsim.timesOff', btn_notsim.timesOff);
    psychoJS.experiment.addData('btn_sim.numClicks', btn_sim.numClicks);
    psychoJS.experiment.addData('btn_sim.timesOn', btn_sim.timesOn);
    psychoJS.experiment.addData('btn_sim.timesOff', btn_sim.timesOff);
    // the Routine "trial" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var QuestionaireMaxDurationReached;
var QuestionaireMaxDuration;
var QuestionaireComponents;
function QuestionaireRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'Questionaire' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    QuestionaireClock.reset();
    routineTimer.reset();
    QuestionaireMaxDurationReached = false;
    // update component parameters for each repeat
    // =============================================================================
    // BEGIN ROUTINE TAB
    // =============================================================================
    
    // Reset questionnaire variables for this routine
    window.rankingComplete = false;
    window.selectedItems = [];
    window.rankingContainer = null;
    
    // Define the ranking options
    var rankingOptions = [
        "Color and Texture",
        "Contrast and Sharpness",
        "Cleanliness and Distortion", 
        "Composition",
        "Object Identification",
        "Scene Category",
        "Emotional Tone",
        "Story Context",
        "Relative Position of Elements"
    ];
    var rankingDescriptions = {
        "Color and Texture": "Refers to the surface quality and visual texture as well as color palette and hue relationships.",
        "Contrast and Sharpness": "How well-defined the edges and differences in luminance or color are in the image.",
        "Cleanliness and Distortion": "Amount of visual grain or pixel-level artifacts in the image.",
        "Composition": "Arrangement of visual elements including balance, alignment, and framing.",
        "Object Identification": "Ease with which objects in the image can be recognized or named.",
        "Scene Category": "General type or theme of the image (e.g., landscape, indoor, street).",
        "Emotional Tone": "Mood conveyed by the image (e.g., calm, eerie, joyful).",
        "Story Context": "Whether the image suggests a narrative or situation.",
        "Relative Position of Elements": "Spatial relationships between objects or parts of the image."
    };
    // Function to create the ranking interface
    function createRankingInterface() {
        // Create main container
        rankingContainer = document.createElement('div');
        rankingContainer.id = 'rankingContainer';
        rankingContainer.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 90%;
            max-width: 800px;
            height: 80vh;
            background: white;
            border: 2px solid #333;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            font-family: Arial, sans-serif;
            overflow-y: auto;
            z-index: 1000;
        `;
    
        // Create title
        var title = document.createElement('h2');
        title.textContent = 'What Influenced Your Choice?';
        title.style.cssText = `
            margin: 0 0 20px 0;
            text-align: center;
            color: #333;
            font-size: 24px;
        `;
    
        // Create instructions
        var instructions = document.createElement('p');
        instructions.innerHTML = `
            <strong>Instructions:</strong> Please check the boxes below to select the factors that influenced your image similarity judgments, in order of importance. 
            Items will appear in your ranking list in the order you select them. You can uncheck items to remove them.
        `;
        instructions.style.cssText = `
            margin-bottom: 20px;
            color: #555;
            line-height: 1.4;
        `;
    
        // Create checkbox container
        var checkboxContainer = document.createElement('div');
        checkboxContainer.style.cssText = `
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 10px;
            margin-bottom: 30px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        `;
    
        // Create checkboxes
        rankingOptions.forEach(function(option, index) {
            var checkboxDiv = document.createElement('div');
            checkboxDiv.style.cssText = `
                display: flex;
                align-items: center;
                padding: 8px;
                background: white;
                border-radius: 5px;
                border: 1px solid #ddd;
                cursor: pointer;
                transition: background-color 0.2s;
            `;
            
            var checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.id = 'ranking_' + index;
            checkbox.value = option;
            checkbox.style.cssText = `
                margin-right: 10px;
                transform: scale(1.2);
                cursor: pointer;
            `;
            
            var label = document.createElement('label');
            label.textContent = option;
            label.htmlFor = 'ranking_' + index;
            label.style.cssText = `
                cursor: pointer;
                font-size: 14px;
                user-select: none;
            `;
            label.title = rankingDescriptions[option];
            // Add click handlers
            function toggleSelection() {
                if (checkbox.checked) {
                    if (selectedItems.indexOf(option) === -1) {
                        selectedItems.push(option);
                        updateRankingDisplay();
                    }
                } else {
                    var idx = selectedItems.indexOf(option);
                    if (idx > -1) {
                        selectedItems.splice(idx, 1);
                        updateRankingDisplay();
                    }
                }
            }
            
            checkbox.addEventListener('change', toggleSelection);
            checkboxDiv.addEventListener('click', function(e) {
                if (e.target !== checkbox) {
                    checkbox.checked = !checkbox.checked;
                    toggleSelection();
                }
            });
            
            // Hover effects
            checkboxDiv.addEventListener('mouseenter', function() {
                checkboxDiv.style.backgroundColor = '#e3f2fd';
            });
            checkboxDiv.addEventListener('mouseleave', function() {
                checkboxDiv.style.backgroundColor = 'white';
            });
            
            checkboxDiv.appendChild(checkbox);
            checkboxDiv.appendChild(label);
            checkboxContainer.appendChild(checkboxDiv);
        });
    
        // Create ranking display section
        var rankingSection = document.createElement('div');
        rankingSection.style.cssText = `margin-bottom: 20px;`;
    
        var rankingTitle = document.createElement('h3');
        rankingTitle.textContent = 'Your Current Ranking:';
        rankingTitle.style.cssText = `
            margin: 0 0 10px 0;
            color: #333;
            font-size: 18px;
        `;
    
        var rankingList = document.createElement('ol');
        rankingList.id = 'rankingList';
        rankingList.style.cssText = `
            background: #f0f8ff;
            padding: 15px 15px 15px 35px;
            border-radius: 8px;
            border: 2px dashed #2196F3;
            min-height: 100px;
            color: #333;
            font-size: 16px;
            line-height: 1.6;
        `;
    
        // Create submit button
        var submitButton = document.createElement('button');
        submitButton.textContent = 'Continue to Complete';
        submitButton.style.cssText = `
            width: 100%;
            padding: 15px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            transition: background-color 0.3s;
        `;
    
        submitButton.addEventListener('mouseenter', function() {
            submitButton.style.backgroundColor = '#45a049';
        });
        submitButton.addEventListener('mouseleave', function() {
            submitButton.style.backgroundColor = '#4CAF50';
        });
    
        submitButton.addEventListener('click', function() {
            // Store ranking data globally
            window.rankingData = {
                ranking_order: selectedItems.join(','),
                ranking_count: selectedItems.length,
                individual_ranks: {}
            };
            
            // Store individual rankings
            selectedItems.forEach(function(item, index) {
                window.rankingData.individual_ranks['rank_' + (index + 1)] = item;
            });
            
            // Store all factors with their ranks (0 if not selected)
            var allFactors = [
                "Color and Texture", "Contrast and Sharpness", "Cleanliness and Distortion", "Composition", "Object Identification",
                "Scene Category", "Emotional Tone", "Story Context", "Relative Position of Elements"
            ];
            
            allFactors.forEach(function(factor) {
                var rank = selectedItems.indexOf(factor) + 1;
                var key = 'factor_' + factor.replace(/\s+/g, '_').toLowerCase();
                window.rankingData[key] = rank > 0 ? rank : 0;
            });
            
            // Clean up and continue
            if (rankingContainer && rankingContainer.parentNode) {
                rankingContainer.parentNode.removeChild(rankingContainer);
            }
            document.body.style.overflow = 'auto';
            rankingComplete = true;
            continueRoutine = false;
        });
    
        // Function to update ranking display
        window.updateRankingDisplay = function() {
            var rankingListElement = document.getElementById('rankingList');
            if (!rankingListElement) return;
            
            rankingListElement.innerHTML = '';
            
            if (selectedItems.length === 0) {
                var emptyMessage = document.createElement('li');
                emptyMessage.textContent = 'No items selected yet. Check boxes above to add items to your ranking.';
                emptyMessage.style.cssText = `
                    color: #999;
                    font-style: italic;
                    list-style: none;
                `;
                rankingListElement.appendChild(emptyMessage);
            } else {
                selectedItems.forEach(function(item, index) {
                    var listItem = document.createElement('li');
                    listItem.textContent = item;
                    listItem.style.cssText = `
                        margin-bottom: 5px;
                        padding: 5px;
                        background: white;
                        border-radius: 4px;
                        border-left: 4px solid #2196F3;
                    `;
                    rankingListElement.appendChild(listItem);
                });
            }
        };
    
        // Assemble the interface
        rankingContainer.appendChild(title);
        rankingContainer.appendChild(instructions);
        rankingContainer.appendChild(checkboxContainer);
        rankingContainer.appendChild(rankingSection);
        rankingSection.appendChild(rankingTitle);
        rankingSection.appendChild(rankingList);
        rankingContainer.appendChild(submitButton);
    
        // Add to page
        document.body.appendChild(rankingContainer);
        document.body.style.overflow = 'hidden';
    
        // Initialize display
        updateRankingDisplay();
    }
    
    // Create the interface when routine starts
    createRankingInterface();
    psychoJS.experiment.addData('Questionaire.started', globalClock.getTime());
    QuestionaireMaxDuration = null
    // keep track of which components have finished
    QuestionaireComponents = [];
    QuestionaireComponents.push(dummy);
    
    for (const thisComponent of QuestionaireComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


function QuestionaireRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'Questionaire' ---
    // get current time
    t = QuestionaireClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    // =============================================================================
    // EACH FRAME TAB
    // =============================================================================
    
    // Check if ranking is complete
    if (rankingComplete) {
        continueRoutine = false;
    }
    
    // *dummy* updates
    if (t >= 0.0 && dummy.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      dummy.tStart = t;  // (not accounting for frame time here)
      dummy.frameNStart = frameN;  // exact frame index
      
      dummy.setAutoDraw(true);
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of QuestionaireComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function QuestionaireRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'Questionaire' ---
    for (const thisComponent of QuestionaireComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('Questionaire.stopped', globalClock.getTime());
    
    // =============================================================================
    // END ROUTINE TAB
    // =============================================================================
    
    // Clean up if routine ends unexpectedly
    if (rankingContainer && rankingContainer.parentNode) {
        rankingContainer.parentNode.removeChild(rankingContainer);
        document.body.style.overflow = 'auto';
    }
    // the Routine "Questionaire" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset();
    
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


var thank_youMaxDurationReached;
var thank_youMaxDuration;
var thank_youComponents;
function thank_youRoutineBegin(snapshot) {
  return async function () {
    TrialHandler.fromSnapshot(snapshot); // ensure that .thisN vals are up to date
    
    //--- Prepare to start Routine 'thank_you' ---
    t = 0;
    frameN = -1;
    continueRoutine = true; // until we're told otherwise
    thank_youClock.reset(routineTimer.getTime());
    routineTimer.add(2.000000);
    thank_youMaxDurationReached = false;
    // update component parameters for each repeat
    psychoJS.experiment.addData('thank_you.started', globalClock.getTime());
    thank_youMaxDuration = null
    // keep track of which components have finished
    thank_youComponents = [];
    thank_youComponents.push(text_3);
    
    for (const thisComponent of thank_youComponents)
      if ('status' in thisComponent)
        thisComponent.status = PsychoJS.Status.NOT_STARTED;
    return Scheduler.Event.NEXT;
  }
}


var frameRemains;
function thank_youRoutineEachFrame() {
  return async function () {
    //--- Loop for each frame of Routine 'thank_you' ---
    // get current time
    t = thank_youClock.getTime();
    frameN = frameN + 1;// number of completed frames (so 0 is the first frame)
    // update/draw components on each frame
    
    // *text_3* updates
    if (t >= 0.0 && text_3.status === PsychoJS.Status.NOT_STARTED) {
      // keep track of start time/frame for later
      text_3.tStart = t;  // (not accounting for frame time here)
      text_3.frameNStart = frameN;  // exact frame index
      
      text_3.setAutoDraw(true);
    }
    
    frameRemains = 0.0 + 2 - psychoJS.window.monitorFramePeriod * 0.75;// most of one frame period left
    if (text_3.status === PsychoJS.Status.STARTED && t >= frameRemains) {
      text_3.setAutoDraw(false);
    }
    
    // check for quit (typically the Esc key)
    if (psychoJS.experiment.experimentEnded || psychoJS.eventManager.getKeys({keyList:['escape']}).length > 0) {
      return quitPsychoJS('The [Escape] key was pressed. Goodbye!', false);
    }
    
    // check if the Routine should terminate
    if (!continueRoutine) {  // a component has requested a forced-end of Routine
      return Scheduler.Event.NEXT;
    }
    
    continueRoutine = false;  // reverts to True if at least one component still running
    for (const thisComponent of thank_youComponents)
      if ('status' in thisComponent && thisComponent.status !== PsychoJS.Status.FINISHED) {
        continueRoutine = true;
        break;
      }
    
    // refresh the screen if continuing
    if (continueRoutine && routineTimer.getTime() > 0) {
      return Scheduler.Event.FLIP_REPEAT;
    } else {
      return Scheduler.Event.NEXT;
    }
  };
}


function thank_youRoutineEnd(snapshot) {
  return async function () {
    //--- Ending Routine 'thank_you' ---
    for (const thisComponent of thank_youComponents) {
      if (typeof thisComponent.setAutoDraw === 'function') {
        thisComponent.setAutoDraw(false);
      }
    }
    psychoJS.experiment.addData('thank_you.stopped', globalClock.getTime());
    if (thank_youMaxDurationReached) {
        thank_youClock.add(thank_youMaxDuration);
    } else {
        thank_youClock.add(2.000000);
    }
    // Routines running outside a loop should always advance the datafile row
    if (currentLoop === psychoJS.experiment) {
      psychoJS.experiment.nextEntry(snapshot);
    }
    return Scheduler.Event.NEXT;
  }
}


function importConditions(currentLoop) {
  return async function () {
    psychoJS.importAttributes(currentLoop.getCurrentTrial());
    return Scheduler.Event.NEXT;
    };
}


async function quitPsychoJS(message, isCompleted) {
  // Check for and save orphaned data
  if (psychoJS.experiment.isEntryEmpty()) {
    psychoJS.experiment.nextEntry();
  }
  // Run 'End Experiment' code from code
  ///////////////////////////
  // "End Experiment" TAB
  ///////////////////////////
  
  // Write out all trial‐wise data
  for (let i = 0; i < shownRows.length; i++) {
    psychoJS.experiment.addData("staircase", stairIndices[i]);
    psychoJS.experiment.addData("original_image",    shownRows[i].original_image);
    psychoJS.experiment.addData("generated_image",   shownRows[i].generated_image);
    psychoJS.experiment.addData("clip_similarity",   shownRows[i].clip_similarity);
    psychoJS.experiment.addData("response",          responses[i]);
    psychoJS.experiment.addData("reaction_time",     reaction_times[i]);
    psychoJS.experiment.addData("key_pressed",       key_presses[i]);
    psychoJS.experiment.addData("original_left",     orig_lefts[i]);
    psychoJS.experiment.nextEntry();
  }
  
  // Add the questionnaire/ranking summary if present
  if (window.rankingData) {
    psychoJS.experiment.addData("trial_type",    "ranking_summary");
    psychoJS.experiment.addData("ranking_order", window.rankingData.ranking_order);
    psychoJS.experiment.addData("ranking_count", window.rankingData.ranking_count);
  
    Object.keys(window.rankingData).forEach(key => {
      if (key.startsWith('factor_')) {
        psychoJS.experiment.addData(key, window.rankingData[key]);
      }
    });
    psychoJS.experiment.nextEntry();
  }
  
  psychoJS.window.close();
  psychoJS.quit({message: message, isCompleted: isCompleted});
  
  return Scheduler.Event.QUIT;
}
