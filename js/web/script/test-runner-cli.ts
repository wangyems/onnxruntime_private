// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/* eslint-disable guard-for-in */
/* eslint-disable @typescript-eslint/no-use-before-define */

import {spawnSync} from 'child_process';
import * as fs from 'fs-extra';
import {default as minimatch} from 'minimatch';
import npmlog from 'npmlog';
import * as os from 'os';
import * as path from 'path';
import {inspect} from 'util';

import {bufferToBase64} from '../test/test-shared';
import {Test} from '../test/test-types';

import {parseTestRunnerCliArgs, TestRunnerCliArgs} from './test-runner-cli-args';

async function main() {
  // use dynamic import so that we can use ESM only libraries in commonJS.
  const {globbySync} = await import('globby');
  const stripJsonComments = (await import('strip-json-comments')).default;

  npmlog.info('TestRunnerCli', 'Initializing...');

  const args = parseTestRunnerCliArgs(process.argv.slice(2));

  npmlog.verbose('TestRunnerCli.Init.Config', inspect(args));

  const TEST_ROOT = path.join(__dirname, '..', 'test');
  const TEST_DATA_MODEL_NODE_ROOT = path.join(TEST_ROOT, 'data', 'node');
  const TEST_DATA_OP_ROOT = path.join(TEST_ROOT, 'data', 'ops');

  const TEST_DATA_BASE = args.env === 'node' ? TEST_ROOT : '/base/test/';

  npmlog.verbose('TestRunnerCli.Init', 'Ensure test data folder...');
  fs.ensureSymlinkSync(path.join(__dirname, '../../test/data/node'), TEST_DATA_MODEL_NODE_ROOT, 'junction');
  npmlog.verbose('TestRunnerCli.Init', 'Ensure test data folder... DONE');

  let testlist: Test.TestList;
  const shouldLoadSuiteTestData = (args.mode === 'suite0' || args.mode === 'suite1');
  if (shouldLoadSuiteTestData) {
    npmlog.verbose('TestRunnerCli.Init', 'Loading testlist...');

    // The following is a list of unittests for already implemented operators.
    // Modify this list to control what node tests to run.
    const jsonWithComments = fs.readFileSync(path.resolve(TEST_ROOT, './suite-test-list.jsonc')).toString();
    const json = stripJsonComments(jsonWithComments, {whitespace: true});
    testlist = JSON.parse(json) as Test.TestList;
    npmlog.verbose('TestRunnerCli.Init', 'Loading testlist... DONE');
  }

  // The default backends and opset version lists. Those will be used in suite tests.
  const DEFAULT_BACKENDS: readonly TestRunnerCliArgs.Backend[] =
      args.env === 'node' ? ['cpu', 'wasm'] : ['wasm', 'webgl', 'webgpu', 'webnn'];
  const DEFAULT_OPSET_VERSIONS = fs.readdirSync(TEST_DATA_MODEL_NODE_ROOT, {withFileTypes: true})
                                     .filter(dir => dir.isDirectory() && dir.name.startsWith('opset'))
                                     .map(dir => dir.name.slice(5));
  const MAX_OPSET_VERSION = Math.max(...DEFAULT_OPSET_VERSIONS.map(v => Number.parseInt(v, 10)));

  const FILE_CACHE_ENABLED = args.fileCache;         // whether to enable file cache
  const FILE_CACHE_MAX_FILE_SIZE = 1 * 1024 * 1024;  // The max size of the file that will be put into file cache
  const FILE_CACHE_SPLIT_SIZE = 4 * 1024 * 1024;     // The min size of the cache file
  const fileCache: Test.FileCache = {};

  const nodeTests = new Map<string, Test.ModelTestGroup[]>();
  const onnxTests = new Map<string, Test.ModelTestGroup>();
  const opTests = new Map<string, Test.OperatorTestGroup[]>();

  if (shouldLoadSuiteTestData) {
    npmlog.verbose('TestRunnerCli.Init', 'Loading test groups for suite test...');

    // collect all model test folders
    const allNodeTestsFolders =
        DEFAULT_OPSET_VERSIONS
            .map(version => {
              const suiteRootFolder = path.join(TEST_DATA_MODEL_NODE_ROOT, `opset${version}`);
              if (!fs.existsSync(suiteRootFolder) || !fs.statSync(suiteRootFolder).isDirectory()) {
                throw new Error(`model test root folder '${suiteRootFolder}' does not exist.`);
              }
              return fs.readdirSync(suiteRootFolder).map(f => `opset${version}/${f}`);
            })
            .flat();

    for (const backend of DEFAULT_BACKENDS) {
      if (args.backends.indexOf(backend) !== -1) {
        nodeTests.set(backend, loadNodeTests(backend, allNodeTestsFolders));
        opTests.set(backend, loadOpTests(backend));
      }
    }
  }

  if (shouldLoadSuiteTestData) {
    npmlog.verbose('TestRunnerCli.Init', 'Loading test groups for suite test... DONE');

    npmlog.verbose('TestRunnerCli.Init', 'Validate testlist...');
    validateTestList();
    npmlog.verbose('TestRunnerCli.Init', 'Validate testlist... DONE');
  }

  const modelTestGroups: Test.ModelTestGroup[] = [];
  const opTestGroups: Test.OperatorTestGroup[] = [];
  let unittest = false;

  npmlog.verbose('TestRunnerCli.Init', 'Preparing test config...');
  switch (args.mode) {
    case 'suite0':
    case 'suite1':
      for (const backend of DEFAULT_BACKENDS) {
        if (args.backends.indexOf(backend) !== -1) {
          modelTestGroups.push(...nodeTests.get(backend)!);  // model test : node
          opTestGroups.push(...opTests.get(backend)!);       // operator test
        }
      }
      if (args.mode === 'suite0') {
        unittest = true;
      }
      break;

    case 'model':
      if (!args.param) {
        throw new Error('the test folder should be specified in mode \'node\'');
      } else {
        const testFolderSearchPattern = args.param;
        const testFolder = tryLocateModelTestFolder(testFolderSearchPattern);
        for (const b of args.backends) {
          modelTestGroups.push({name: testFolder, tests: [modelTestFromFolder(testFolder, b, undefined, args.times)]});
        }
      }
      break;

    case 'unittest':
      unittest = true;
      break;

    case 'op':
      if (!args.param) {
        throw new Error('the test manifest should be specified in mode \'op\'');
      } else {
        const manifestFileSearchPattern = args.param;
        const manifestFile = tryLocateOpTestManifest(manifestFileSearchPattern);
        for (const b of args.backends) {
          opTestGroups.push(opTestFromManifest(manifestFile, b));
        }
      }
      break;
    default:
      throw new Error(`unsupported mode '${args.mode}'`);
  }

  npmlog.verbose('TestRunnerCli.Init', 'Preparing test config... DONE');

  npmlog.info('TestRunnerCli', 'Initialization completed. Start to run tests...');
  run({
    unittest,
    model: modelTestGroups,
    op: opTestGroups,
    log: args.logConfig,
    profile: args.profile,
    options: {
      sessionOptions:
          {graphOptimizationLevel: args.graphOptimizationLevel, optimizedModelFilePath: args.optimizedModelFilePath},
      debug: args.debug,
      cpuOptions: args.cpuOptions,
      webglOptions: args.webglOptions,
      wasmOptions: args.wasmOptions,
      globalEnvFlags: args.globalEnvFlags
    }
  });
  npmlog.info('TestRunnerCli', 'Tests completed successfully');

  function validateTestList() {
    for (const backend of DEFAULT_BACKENDS) {
      const nodeTest = nodeTests.get(backend);
      if (nodeTest) {
        for (const testCase of testlist[backend].node) {
          const testCaseName = typeof testCase === 'string' ? testCase : testCase.name;
          let found = false;
          for (const testGroup of nodeTest) {
            found ||= minimatch
                          .match(
                              testGroup.tests.map(test => test.modelUrl).filter(url => url !== ''),
                              path.join('**', testCaseName, '*.+(onnx|ort)').replace(/\\/g, '/'), {matchBase: true})
                          .length > 0;
          }
          if (!found) {
            throw new Error(`node model test case '${testCaseName}' in test list does not exist.`);
          }
        }
      }

      const onnxTest = onnxTests.get(backend);
      if (onnxTest) {
        const onnxModelTests = onnxTest.tests.map(i => i.name);
        for (const testCase of testlist[backend].onnx) {
          const testCaseName = typeof testCase === 'string' ? testCase : testCase.name;
          if (onnxModelTests.indexOf(testCaseName) === -1) {
            throw new Error(`onnx model test case '${testCaseName}' in test list does not exist.`);
          }
        }
      }

      const opTest = opTests.get(backend);
      if (opTest) {
        const opTests = opTest.map(i => i.name);
        for (const testCase of testlist[backend].ops) {
          const testCaseName = typeof testCase === 'string' ? testCase : testCase.name;
          if (opTests.indexOf(testCaseName) === -1) {
            throw new Error(`operator test case '${testCaseName}' in test list does not exist.`);
          }
        }
      }
    }
  }

  function loadNodeTests(backend: string, allFolders: string[]): Test.ModelTestGroup[] {
    const allTests = testlist[backend]?.node;

    // key is folder name, value is test index array
    const folderTestMatchCount = new Map<string, number[]>(allFolders.map(f => [f, []]));
    // key is test category, value is a list of model test
    const opsetTests = new Map<string, Test.ModelTest[]>();

    allTests.forEach((test, i) => {
      const testName = typeof test === 'string' ? test : test.name;
      const matches = minimatch.match(allFolders, path.join('**', testName).replace(/\\/g, '/'));
      matches.forEach(m => folderTestMatchCount.get(m)!.push(i));
    });

    for (const folder of allFolders) {
      const testIds = folderTestMatchCount.get(folder);
      const times = testIds ? testIds.length : 0;
      if (times > 1) {
        throw new Error(`multiple testlist rules matches test: ${path.join(TEST_DATA_MODEL_NODE_ROOT, folder)}`);
      }

      const test = testIds && testIds.length > 0 ? allTests[testIds[0]] : undefined;
      const platformCondition = test && typeof test !== 'string' ? test.platformCondition : undefined;

      const opsetVersion = folder.split('/')[0];
      const category = `node-${opsetVersion}-${backend}`;
      let modelTests = opsetTests.get(category);
      if (!modelTests) {
        modelTests = [];
        opsetTests.set(category, modelTests);
      }
      modelTests.push(
          modelTestFromFolder(path.resolve(TEST_DATA_MODEL_NODE_ROOT, folder), backend, platformCondition, times));
    }

    return Array.from(opsetTests.keys()).map(category => ({name: category, tests: opsetTests.get(category)!}));
  }

  function modelTestFromFolder(
      testDataRootFolder: string, backend: string, platformCondition?: Test.PlatformCondition,
      times?: number): Test.ModelTest {
    if (times === 0) {
      npmlog.verbose('TestRunnerCli.Init.Model', `Skip test data from folder: ${testDataRootFolder}`);
      return {name: path.basename(testDataRootFolder), backend, modelUrl: '', cases: []};
    }

    let modelUrl: string|null = null;
    let cases: Test.ModelTestCase[] = [];

    npmlog.verbose('TestRunnerCli.Init.Model', `Start to prepare test data from folder: ${testDataRootFolder}`);

    try {
      for (const thisPath of fs.readdirSync(testDataRootFolder)) {
        const thisFullPath = path.join(testDataRootFolder, thisPath);
        const stat = fs.lstatSync(thisFullPath);
        if (stat.isFile()) {
          const ext = path.extname(thisPath);
          if (ext.toLowerCase() === '.onnx' || ext.toLowerCase() === '.ort') {
            if (modelUrl === null) {
              modelUrl = path.join(TEST_DATA_BASE, path.relative(TEST_ROOT, thisFullPath));
              if (FILE_CACHE_ENABLED && !fileCache[modelUrl] && stat.size <= FILE_CACHE_MAX_FILE_SIZE) {
                fileCache[modelUrl] = bufferToBase64(fs.readFileSync(thisFullPath));
              }
            } else {
              throw new Error('there are multiple model files under the folder specified');
            }
          }
        } else if (stat.isDirectory()) {
          const dataFiles: string[] = [];
          for (const dataFile of fs.readdirSync(thisFullPath)) {
            const dataFileFullPath = path.join(thisFullPath, dataFile);
            const ext = path.extname(dataFile);

            if (ext.toLowerCase() === '.pb') {
              const dataFileUrl = path.join(TEST_DATA_BASE, path.relative(TEST_ROOT, dataFileFullPath));
              dataFiles.push(dataFileUrl);
              if (FILE_CACHE_ENABLED && !fileCache[dataFileUrl] &&
                  fs.lstatSync(dataFileFullPath).size <= FILE_CACHE_MAX_FILE_SIZE) {
                fileCache[dataFileUrl] = bufferToBase64(fs.readFileSync(dataFileFullPath));
              }
            }
          }
          if (dataFiles.length > 0) {
            cases.push({dataFiles, name: thisPath});
          }
        }
      }

      if (modelUrl === null) {
        throw new Error('there are no model file under the folder specified');
      }
    } catch (e) {
      npmlog.error('TestRunnerCli.Init.Model', `Failed to prepare test data. Error: ${inspect(e)}`);
      throw e;
    }

    const caseCount = cases.length;
    if (times !== undefined) {
      if (times > caseCount) {
        for (let i = 0; cases.length < times; i++) {
          const origin = cases[i % caseCount];
          const duplicated = {name: `${origin.name} - copy ${Math.floor(i / caseCount)}`, dataFiles: origin.dataFiles};
          cases.push(duplicated);
        }
      } else {
        cases = cases.slice(0, times);
      }
    }

    npmlog.verbose('TestRunnerCli.Init.Model', 'Finished preparing test data.');
    npmlog.verbose('TestRunnerCli.Init.Model', '===============================================================');
    npmlog.verbose('TestRunnerCli.Init.Model', ` Model file: ${modelUrl}`);
    npmlog.verbose('TestRunnerCli.Init.Model', ` Backend: ${backend}`);
    npmlog.verbose('TestRunnerCli.Init.Model', ` Test set(s): ${cases.length} (${caseCount})`);
    npmlog.verbose('TestRunnerCli.Init.Model', '===============================================================');

    return {name: path.basename(testDataRootFolder), platformCondition, modelUrl, backend, cases};
  }

  function tryLocateModelTestFolder(searchPattern: string): string {
    const folderCandidates: string[] = [];
    // 1 - check whether search pattern is a directory
    if (fs.existsSync(searchPattern) && fs.lstatSync(searchPattern).isDirectory()) {
      folderCandidates.push(searchPattern);
    }

    // 2 - check the globby result of searchPattern
    // 3 - check the globby result of ONNX root combined with searchPattern
    const globbyPattern =
        [searchPattern, path.join(TEST_DATA_MODEL_NODE_ROOT, '**', searchPattern).replace(/\\/g, '/')];
    // 4 - check the globby result of NODE root combined with opset versions and searchPattern
    globbyPattern.push(...DEFAULT_OPSET_VERSIONS.map(
        v => path.join(TEST_DATA_MODEL_NODE_ROOT, `opset${v}`, '**', searchPattern).replace(/\\/g, '/')));

    folderCandidates.push(...globbySync(globbyPattern, {onlyDirectories: true, absolute: true}));

    // pick the first folder that matches the pattern
    for (const folderCandidate of folderCandidates) {
      const modelCandidates = globbySync('*.{onnx,ort}', {onlyFiles: true, cwd: folderCandidate});
      if (modelCandidates && modelCandidates.length === 1) {
        return folderCandidate;
      }
    }

    throw new Error(`no model folder found: ${searchPattern}`);
  }

  function loadOpTests(backend: string): Test.OperatorTestGroup[] {
    const groups: Test.OperatorTestGroup[] = [];
    for (const thisPath of fs.readdirSync(TEST_DATA_OP_ROOT)) {
      const thisFullPath = path.join(TEST_DATA_OP_ROOT, thisPath);
      const stat = fs.lstatSync(thisFullPath);
      const ext = path.extname(thisFullPath);
      if (stat.isFile() && (ext === '.json' || ext === '.jsonc')) {
        const skip = testlist[backend].ops.indexOf(thisPath) === -1;
        groups.push(opTestFromManifest(thisFullPath, backend, skip));
      }
    }

    return groups;
  }

  function opTestFromManifest(manifestFile: string, backend: string, skip = false): Test.OperatorTestGroup {
    let tests: Test.OperatorTest[] = [];
    const filePath = path.resolve(process.cwd(), manifestFile);
    if (skip) {
      npmlog.verbose('TestRunnerCli.Init.Op', `Skip test data from manifest file: ${filePath}`);
    } else {
      npmlog.verbose('TestRunnerCli.Init.Op', `Start to prepare test data from manifest file: ${filePath}`);
      const jsonWithComments = fs.readFileSync(filePath).toString();
      const json = stripJsonComments(jsonWithComments, {whitespace: true});
      tests = JSON.parse(json) as Test.OperatorTest[];
      // field 'verbose' and 'backend' is not set
      for (const test of tests) {
        test.backend = backend;
        test.opset = test.opset || {domain: '', version: MAX_OPSET_VERSION};
      }
      npmlog.verbose('TestRunnerCli.Init.Op', 'Finished preparing test data.');
      npmlog.verbose('TestRunnerCli.Init.Op', '===============================================================');
      npmlog.verbose('TestRunnerCli.Init.Op', ` Test Group: ${path.relative(TEST_DATA_OP_ROOT, filePath)}`);
      npmlog.verbose('TestRunnerCli.Init.Op', ` Backend: ${backend}`);
      npmlog.verbose('TestRunnerCli.Init.Op', ` Test case(s): ${tests.length}`);
      npmlog.verbose('TestRunnerCli.Init.Op', '===============================================================');
    }
    return {name: path.relative(TEST_DATA_OP_ROOT, filePath), tests};
  }

  function tryLocateOpTestManifest(searchPattern: string): string {
    for (const manifestCandidate of globbySync(
             [
               searchPattern, path.join(TEST_DATA_OP_ROOT, '**', searchPattern).replace(/\\/g, '/'),
               path.join(TEST_DATA_OP_ROOT, '**', searchPattern + '.json').replace(/\\/g, '/'),
               path.join(TEST_DATA_OP_ROOT, '**', searchPattern + '.jsonc').replace(/\\/g, '/')
             ],
             {onlyFiles: true, absolute: true, cwd: TEST_ROOT})) {
      return manifestCandidate;
    }

    throw new Error(`no OP test manifest found: ${searchPattern}`);
  }

  function run(config: Test.Config) {
    // STEP 1. write file cache to testdata-file-cache-*.json
    npmlog.info('TestRunnerCli.Run', '(1/4) Writing file cache to file: testdata-file-cache-*.json ...');
    const fileCacheUrls = saveFileCache(fileCache);
    if (fileCacheUrls.length > 0) {
      config.fileCacheUrls = fileCacheUrls;
    }
    npmlog.info(
        'TestRunnerCli.Run',
        `(1/4) Writing file cache to file: testdata-file-cache-*.json ... ${
            fileCacheUrls.length > 0 ? `DONE, ${fileCacheUrls.length} file(s) generated` : 'SKIPPED'}`);

    // STEP 2. write the config to testdata-config.json
    npmlog.info('TestRunnerCli.Run', '(2/4) Writing config to file: testdata-config.json ...');
    saveConfig(config);
    npmlog.info('TestRunnerCli.Run', '(2/4) Writing config to file: testdata-config.json ... DONE');

    // STEP 3. generate bundle
    npmlog.info('TestRunnerCli.Run', '(3/4) Running build to generate bundle...');
    const buildCommand = `node ${path.join(__dirname, 'build')}`;
    const buildArgs = [`--bundle-mode=${args.env === 'node' ? 'node' : args.bundleMode}`];
    if (args.backends.indexOf('wasm') === -1) {
      buildArgs.push('--no-wasm');
    }
    npmlog.info('TestRunnerCli.Run', `CMD: ${buildCommand} ${buildArgs.join(' ')}`);
    const build = spawnSync(buildCommand, buildArgs, {shell: true, stdio: 'inherit'});
    if (build.status !== 0) {
      console.error(build.error);
      process.exit(build.status === null ? undefined : build.status);
    }
    npmlog.info('TestRunnerCli.Run', '(3/4) Running build to generate bundle... DONE');

    if (args.env === 'node') {
      // STEP 5. run tsc and run mocha
      npmlog.info('TestRunnerCli.Run', '(4/4) Running tsc...');
      const tsc = spawnSync('npx', ['tsc'], {shell: true, stdio: 'inherit'});
      if (tsc.status !== 0) {
        console.error(tsc.error);
        process.exit(tsc.status === null ? undefined : tsc.status);
      }
      npmlog.info('TestRunnerCli.Run', '(4/4) Running tsc... DONE');

      npmlog.info('TestRunnerCli.Run', '(4/4) Running mocha...');
      const mochaArgs = ['mocha', path.join(TEST_ROOT, 'test-main'), `--timeout ${args.debug ? 9999999 : 60000}`];
      npmlog.info('TestRunnerCli.Run', `CMD: npx ${mochaArgs.join(' ')}`);
      const mocha = spawnSync('npx', mochaArgs, {shell: true, stdio: 'inherit'});
      if (mocha.status !== 0) {
        console.error(mocha.error);
        process.exit(mocha.status === null ? undefined : mocha.status);
      }
      npmlog.info('TestRunnerCli.Run', '(4/4) Running mocha... DONE');

    } else {
      // STEP 5. use Karma to run test
      npmlog.info('TestRunnerCli.Run', '(4/4) Running karma to start test runner...');
      const webgpu = args.backends.indexOf('webgpu') > -1;
      const webnn = args.backends.indexOf('webnn') > -1;
      const browser = getBrowserNameFromEnv(
          args.env,
          args.bundleMode === 'perf' ? 'perf' :
              args.debug             ? 'debug' :
                                       'test',
          webgpu, webnn);
      const karmaArgs = ['karma', 'start', `--browsers ${browser}`];
      const chromiumFlags = ['--enable-features=SharedArrayBuffer', ...args.chromiumFlags];
      if (args.debug) {
        karmaArgs.push('--log-level info --timeout-mocha 9999999');
        chromiumFlags.push('--remote-debugging-port=9333');
      } else {
        karmaArgs.push('--single-run');
      }
      if (args.noSandbox) {
        karmaArgs.push('--no-sandbox');
      }
      if (webgpu || webnn) {
        karmaArgs.push('--force-localhost');
      }
      if (webgpu) {
        if (browser.includes('Canary')) {
          chromiumFlags.push('--enable-dawn-features=allow_unsafe_apis,use_dxc');
        } else {
          chromiumFlags.push('--enable-dawn-features=use_dxc');
          chromiumFlags.push('--disable-dawn-features=disallow_unsafe_apis');
        }
      }
      if (webnn) {
        chromiumFlags.push('--enable-experimental-web-platform-features');
      }
      if (config.options.globalEnvFlags?.webgpu?.profilingMode === 'default') {
        chromiumFlags.push('--disable-dawn-features=disallow_unsafe_apis');
      }
      karmaArgs.push(`--bundle-mode=${args.bundleMode}`);
      karmaArgs.push(...chromiumFlags.map(flag => `--chromium-flags=${flag}`));
      if (browser.startsWith('Edge')) {
        // There are currently 2 Edge browser launchers:
        //  - karma-edge-launcher: used to launch the old Edge browser
        //  - karma-chromium-edge-launcher: used to launch the new chromium-kernel Edge browser
        //

        // Those 2 plugins cannot be loaded at the same time, so we need to determine which launchers to use.
        //  - If we use 'karma-edge-launcher', no plugins config need to be set.
        //  - If we use 'karma-chromium-edge-launcher', we need to:
        //       - add plugin "@chiragrupani/karma-chromium-edge-launcher" explicitly, because it does not match the
        //         default plugins config "^karma-.*"
        //       - remove "karma-edge-launcher".

        // check if we have the latest Edge installed:
        if (os.platform() === 'darwin' ||
            (os.platform() === 'win32' &&
             require('@chiragrupani/karma-chromium-edge-launcher/dist/Utilities').default.GetEdgeExe('Edge') !== '')) {
          // use "@chiragrupani/karma-chromium-edge-launcher"
          karmaArgs.push(
              '--karma-plugins=@chiragrupani/karma-chromium-edge-launcher',
              '--karma-plugins=(?!karma-edge-launcher$)karma-*');
        } else {
          // use "karma-edge-launcher"

          // == Special treatment to Microsoft Edge ==
          //
          // == Edge's Auto Recovery Feature ==
          // when Edge starts, if it found itself was terminated forcely last time, it always recovers all previous
          // pages. this always happen in Karma because `karma-edge-launcher` uses `taskkill` command to kill Edge every
          // time.
          //
          // == The Problem ==
          // every time when a test is completed, it will be added to the recovery page list.
          // if we run the test 100 times, there will be 100 previous tabs when we launch Edge again.
          // this run out of resources quickly and fails the futher test.
          // and it cannot recover by itself because every time it is terminated forcely or crashes.
          // and the auto recovery feature has no way to disable by configuration/commandline/registry
          //
          // == The Solution ==
          // for Microsoft Edge, we should clean up the previous active page before each run
          // delete the files stores in the specific folder to clean up the recovery page list.
          // see also: https://www.laptopmag.com/articles/edge-browser-stop-tab-restore
          const deleteEdgeActiveRecoveryCommand =
              // eslint-disable-next-line max-len
              'del /F /Q % LOCALAPPDATA %\\Packages\\Microsoft.MicrosoftEdge_8wekyb3d8bbwe\\AC\\MicrosoftEdge\\User\\Default\\Recovery\\Active\\*';
          npmlog.info('TestRunnerCli.Run', `CMD: ${deleteEdgeActiveRecoveryCommand}`);
          spawnSync(deleteEdgeActiveRecoveryCommand, {shell: true, stdio: 'inherit'});
        }
      }
      npmlog.info('TestRunnerCli.Run', `CMD: npx ${karmaArgs.join(' ')}`);
      const karma = spawnSync('npx', karmaArgs, {shell: true, stdio: 'inherit'});
      if (karma.status !== 0) {
        console.error(karma.error);
        process.exit(karma.status === null ? undefined : karma.status);
      }
      npmlog.info('TestRunnerCli.Run', '(4/4) Running karma to start test runner... DONE');
    }
  }

  function saveFileCache(fileCache: Test.FileCache) {
    const fileCacheUrls: string[] = [];
    let currentIndex = 0;
    let currentCache: Test.FileCache = {};
    let currentContentTotalSize = 0;
    for (const key in fileCache) {
      const content = fileCache[key];
      if (currentContentTotalSize > FILE_CACHE_SPLIT_SIZE) {
        fileCacheUrls.push(saveOneFileCache(currentIndex, currentCache));
        currentContentTotalSize = 0;
        currentIndex++;
        currentCache = {};
      }
      currentCache[key] = content;
      currentContentTotalSize += key.length + content.length;
    }
    if (currentContentTotalSize > 0) {
      fileCacheUrls.push(saveOneFileCache(currentIndex, currentCache));
    }
    return fileCacheUrls;
  }

  function saveOneFileCache(index: number, fileCache: Test.FileCache) {
    fs.writeFileSync(path.join(TEST_ROOT, `./testdata-file-cache-${index}.json`), JSON.stringify(fileCache));
    return path.join(TEST_DATA_BASE, `./testdata-file-cache-${index}.json`);
  }

  function saveConfig(config: Test.Config) {
    fs.writeJSONSync(path.join(TEST_ROOT, './testdata-config.json'), config);
  }

  function getBrowserNameFromEnv(
      env: TestRunnerCliArgs['env'], mode: 'debug'|'perf'|'test', webgpu: boolean, webnn: boolean) {
    switch (env) {
      case 'chrome':
        return selectChromeBrowser(mode, webgpu, webnn);
      case 'edge':
        return 'EdgeTest';
      case 'firefox':
        return 'Firefox';
      case 'electron':
        return 'Electron';
      case 'safari':
        return 'Safari';
      case 'bs':
        return process.env.ORT_WEB_TEST_BS_BROWSERS!;
      default:
        throw new Error(`env "${env}" not supported.`);
    }
  }

  function selectChromeBrowser(mode: 'debug'|'perf'|'test', webgpu: boolean, webnn: boolean) {
    if (webnn) {
      return 'ChromeCanaryTest';
    } else if (webgpu) {
      return 'ChromeTest';
    } else {
      switch (mode) {
        case 'debug':
        case 'perf':
          return 'ChromeTest';
        default:
          return 'ChromeTestHeadless';
      }
    }
  }
}

void main();
