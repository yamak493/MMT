// 3D プレビュー（Babylon.js ＋ babylon-mmd）。モデルを表示し、タイムラインから作ったモーションを再生する。
import "@babylonjs/core/Loading/loadingScreen";
import { ArcRotateCamera } from "@babylonjs/core/Cameras/arcRotateCamera";
import { Engine } from "@babylonjs/core/Engines/engine";
import { DirectionalLight } from "@babylonjs/core/Lights/directionalLight";
import { HemisphericLight } from "@babylonjs/core/Lights/hemisphericLight";
import { ShadowGenerator } from "@babylonjs/core/Lights/Shadows/shadowGenerator";
import { LoadAssetContainerAsync } from "@babylonjs/core/Loading/sceneLoader";
import { Color3, Color4 } from "@babylonjs/core/Maths/math.color";
import { Vector3 } from "@babylonjs/core/Maths/math.vector";
import type { AssetContainer } from "@babylonjs/core/assetContainer";
import type { Mesh } from "@babylonjs/core/Meshes/mesh";
import { CreateGround } from "@babylonjs/core/Meshes/Builders/groundBuilder";
import { CreateLineSystem } from "@babylonjs/core/Meshes/Builders/linesBuilder";
import type { LinesMesh } from "@babylonjs/core/Meshes/linesMesh";
import { Scene } from "@babylonjs/core/scene";
import { StandardMaterial } from "@babylonjs/core/Materials/standardMaterial";

import "babylon-mmd/esm/Loader/pmxLoader";
import "babylon-mmd/esm/Loader/pmdLoader";
import "babylon-mmd/esm/Loader/mmdOutlineRenderer";
import "babylon-mmd/esm/Runtime/Animation/mmdRuntimeModelAnimation";
import { MmdAnimation } from "babylon-mmd/esm/Loader/Animation/mmdAnimation";
import {
    MmdBoneAnimationTrack, MmdCameraAnimationTrack, MmdMorphAnimationTrack, MmdMovableBoneAnimationTrack, MmdPropertyAnimationTrack
} from "babylon-mmd/esm/Loader/Animation/mmdAnimationTrack";
import { MmdStandardMaterialBuilder } from "babylon-mmd/esm/Loader/mmdStandardMaterialBuilder";
import { PmdReader } from "babylon-mmd/esm/Loader/Parser/pmdReader";
import type { PmxObject } from "babylon-mmd/esm/Loader/Parser/pmxObject";
import { PmxReader } from "babylon-mmd/esm/Loader/Parser/pmxReader";
import { RegisterDxBmpTextureLoader } from "babylon-mmd/esm/Loader/registerDxBmpTextureLoader";
import { SdefInjector } from "babylon-mmd/esm/Loader/sdefInjector";
import type { IPlayer } from "babylon-mmd/esm/Runtime/Audio/IAudioPlayer";
import type { MmdModel } from "babylon-mmd/esm/Runtime/mmdModel";
import type { MmdRuntimeAnimationHandle } from "babylon-mmd/esm/Runtime/mmdRuntimeAnimationHandle";
import { MmdRuntime } from "babylon-mmd/esm/Runtime/mmdRuntime";
import { MmdStandardMaterialProxy } from "babylon-mmd/esm/Runtime/mmdStandardMaterialProxy";
import { MmdWasmInstanceTypeSPR } from "babylon-mmd/esm/Runtime/Optimized/InstanceType/singlePhysicsRelease";
import { GetMmdWasmInstance } from "babylon-mmd/esm/Runtime/Optimized/mmdWasmInstance";
import { MultiPhysicsRuntime } from "babylon-mmd/esm/Runtime/Optimized/Physics/Bind/Impl/multiPhysicsRuntime";
import { MmdBulletPhysics } from "babylon-mmd/esm/Runtime/Optimized/Physics/mmdBulletPhysics";
import type { ComposedMotion } from "../motion/keys";
import { MMD_FPS } from "../../shared/types";

RegisterDxBmpTextureLoader();

export interface LoadedModel {
    object: PmxObject;
    name: string;
}

export class Viewer {
    readonly engine: Engine;
    readonly scene: Scene;
    readonly camera: ArcRotateCamera;
    private runtime!: MmdRuntime;
    private physicsRuntime: MultiPhysicsRuntime | null = null;
    private shadow: ShadowGenerator;
    private container: AssetContainer | null = null;
    private model: MmdModel | null = null;
    private animationHandle: MmdRuntimeAnimationHandle | null = null;
    private grid: LinesMesh;
    private resizeObserver: ResizeObserver;
    private lastMotion: ComposedMotion | null = null;
    private manualDuration = 0;
    physicsEnabled = true;
    physicsAvailable = false;
    /** 毎フレーム（描画の前）に呼ばれる */
    onTick: ((frame: number, playing: boolean) => void) | null = null;

    constructor(canvas: HTMLCanvasElement) {
        this.engine = new Engine(canvas, true, { preserveDrawingBuffer: false, stencil: true, antialias: true }, true);
        SdefInjector.OverrideEngineCreateEffect(this.engine);
        const scene = (this.scene = new Scene(this.engine));
        scene.clearColor = new Color4(0.13, 0.14, 0.16, 1);
        scene.ambientColor = new Color3(0.5, 0.5, 0.5);

        const camera = (this.camera = new ArcRotateCamera("camera", -Math.PI / 2, 1.35, 38, new Vector3(0, 10, 0), scene));
        camera.minZ = 0.5;
        camera.maxZ = 2000;
        camera.lowerRadiusLimit = 2;
        camera.upperRadiusLimit = 400;
        camera.wheelDeltaPercentage = 0.02;
        camera.panningSensibility = 60;
        camera.attachControl(canvas, true);
        camera.inputs.attached.keyboard?.detachControl();
        // 左ドラッグで視点の移動（パン）、右ドラッグで角度の調整（回転）。マウスの割り当ては先に一致したものが使われる
        const inputMap = camera.movement.input.inputMap;
        for (let i = inputMap.length - 1; i >= 0; i--) {
            if (inputMap[i].source === "pointer" && "button" in inputMap[i]) inputMap.splice(i, 1);
        }
        inputMap.unshift(
            { source: "pointer", button: 0, interaction: "pan" },
            { source: "pointer", button: 2, interaction: "rotate" }
        );

        const hemi = new HemisphericLight("hemi", new Vector3(0, 1, 0), scene);
        hemi.intensity = 0.35;
        hemi.groundColor = new Color3(0.3, 0.3, 0.32);
        const sun = new DirectionalLight("sun", new Vector3(0.5, -1, 1), scene);
        sun.intensity = 0.75;
        sun.position = new Vector3(-20, 40, -30);
        sun.autoCalcShadowZBounds = true;
        this.shadow = new ShadowGenerator(2048, sun, true);
        this.shadow.usePercentageCloserFiltering = true;
        this.shadow.forceBackFacesOnly = true;
        this.shadow.bias = 0.004;
        this.shadow.normalBias = 0.02;

        const ground = CreateGround("ground", { width: 120, height: 120 }, scene);
        const groundMat = new StandardMaterial("ground", scene);
        groundMat.diffuseColor = new Color3(0.2, 0.21, 0.24);
        groundMat.specularColor = new Color3(0, 0, 0);
        groundMat.ambientColor = new Color3(0.2, 0.21, 0.24);
        ground.material = groundMat;
        ground.receiveShadows = true;
        this.grid = this.createGrid();

        this.runtime = new MmdRuntime(scene);
        this.runtime.register(scene);
        this.runtime.loggingEnabled = false;

        this.engine.runRenderLoop(() => {
            this.onTick?.(this.runtime.currentFrameTime, this.runtime.isAnimationPlaying);
            scene.render();
        });
        this.resizeObserver = new ResizeObserver(() => this.engine.resize());
        this.resizeObserver.observe(canvas);
    }

    /** 物理演算（Bullet の WebAssembly）を準備する。読めなければ物理演算なしで表示する。 */
    async initPhysics(): Promise<void> {
        try {
            const wasm = await GetMmdWasmInstance(new MmdWasmInstanceTypeSPR());
            const physicsRuntime = new MultiPhysicsRuntime(wasm);
            physicsRuntime.setGravity(new Vector3(0, -98, 0));
            physicsRuntime.register(this.scene);
            this.physicsRuntime = physicsRuntime;
            const old = this.runtime;
            this.runtime = new MmdRuntime(this.scene, new MmdBulletPhysics(physicsRuntime));
            this.runtime.loggingEnabled = false;
            old.unregister(this.scene);
            old.dispose(this.scene);
            this.runtime.register(this.scene);
            this.physicsAvailable = true;
        } catch (e) {
            console.warn("物理演算を準備できませんでした", e);
            this.physicsAvailable = false;
        }
    }

    private createGrid(): LinesMesh {
        const lines: Vector3[][] = [];
        const colors: Color4[][] = [];
        const size = 50;
        const step = 5;
        for (let i = -size; i <= size; i += step) {
            const c = i === 0 ? new Color4(0.55, 0.58, 0.62, 1) : new Color4(0.3, 0.32, 0.36, 1);
            lines.push([new Vector3(i, 0.01, -size), new Vector3(i, 0.01, size)]);
            lines.push([new Vector3(-size, 0.01, i), new Vector3(size, 0.01, i)]);
            colors.push([c, c], [c, c]);
        }
        const grid = CreateLineSystem("grid", { lines, colors }, this.scene);
        grid.isPickable = false;
        return grid;
    }

    set gridVisible(v: boolean) {
        this.grid.setEnabled(v);
    }

    /** モデル（PMX / PMD）を読み込む。戻り値はパースしたモデル（リグ情報を作るのに使う）。 */
    async loadModel(path: string, url: string): Promise<LoadedModel> {
        const ext = path.toLowerCase().endsWith(".pmd") ? ".pmd" : ".pmx";
        const data = await window.mmt.readFile(path);
        const object = ext === ".pmd" ? await PmdReader.ParseAsync(data) : await PmxReader.ParseAsync(data);
        this.unloadModel();
        const container = await LoadAssetContainerAsync(url, this.scene, {
            pluginExtension: ext,
            pluginOptions: { mmdmodel: { materialBuilder: new MmdStandardMaterialBuilder(), loggingEnabled: false } }
        });
        container.addAllToScene();
        this.container = container;
        this.createModel();
        this.focusModel();
        return { object, name: object.header.modelName || path.split(/[\\/]/).pop()! };
    }

    private createModel(): void {
        if (!this.container) return;
        const mesh = this.container.meshes[0] as Mesh;
        for (const m of this.container.meshes) {
            this.shadow.addShadowCaster(m);
            m.receiveShadows = true;
        }
        this.model = this.runtime.createMmdModel(mesh, {
            materialProxyConstructor: MmdStandardMaterialProxy,
            buildPhysics: this.physicsEnabled && this.physicsAvailable,
            // 物理演算の入り切りでモデルを作り直すので、メッシュの MMD のメタデータを残す
            trimMetadata: false
        });
        if (this.lastMotion) this.setMotion(this.lastMotion);
    }

    unloadModel(): void {
        if (this.model) {
            this.runtime.destroyMmdModel(this.model);
            this.model = null;
            this.animationHandle = null;
        }
        if (this.container) {
            this.container.removeAllFromScene();
            this.container.dispose();
            this.container = null;
        }
    }

    get hasModel(): boolean {
        return this.model !== null;
    }

    /** 物理演算の入り切り（モデルを作り直す） */
    setPhysics(enabled: boolean): void {
        this.physicsEnabled = enabled;
        if (!this.container || !this.model) return;
        const frame = this.runtime.currentFrameTime;
        this.runtime.destroyMmdModel(this.model);
        this.model = null;
        this.animationHandle = null;
        this.createModel();
        void this.runtime.seekAnimation(frame, true);
    }

    focusModel(): void {
        const mesh = this.container?.meshes[0];
        if (!mesh) return;
        const { min, max } = mesh.getHierarchyBoundingVectors(true);
        const height = Math.max(1, max.y - min.y);
        this.camera.setTarget(new Vector3((min.x + max.x) / 2, min.y + height * 0.55, (min.z + max.z) / 2));
        this.camera.alpha = -Math.PI / 2;
        this.camera.beta = 1.38;
        this.camera.radius = height * 2.1;
    }

    /** モーションを掛ける（null で外す）。 */
    setMotion(motion: ComposedMotion | null): void {
        this.lastMotion = motion;
        if (!this.model) return;
        const prev = this.animationHandle;
        if (!motion || (!motion.bones.length && !motion.morphs.length)) {
            this.model.setRuntimeAnimation(null);
            if (prev) this.model.destroyRuntimeAnimation(prev);
            this.animationHandle = null;
        } else {
            const handle = this.model.createRuntimeAnimation(toMmdAnimation(motion));
            this.model.setRuntimeAnimation(handle);
            if (prev) this.model.destroyRuntimeAnimation(prev);
            this.animationHandle = handle;
        }
        this.updateDuration();
        void this.runtime.seekAnimation(this.runtime.currentFrameTime, true);
    }

    /** 再生の時刻を決めるもの（動画）。null なら内部の時計で再生する（長さは setDuration）。 */
    async setPlayer(player: IPlayer | null): Promise<void> {
        await this.runtime.setAudioPlayer(player);
        this.updateDuration();
    }

    /** 動画が無いときのタイムラインの長さ [フレーム] */
    setDuration(frames: number): void {
        this.manualDuration = frames;
        this.updateDuration();
    }

    private updateDuration(): void {
        if (this.runtime.audioPlayer) {
            this.runtime.setManualAnimationDuration(null);
        } else {
            const last = this.lastMotion?.lastFrame ?? 0;
            this.runtime.setManualAnimationDuration(Math.max(this.manualDuration, last, 1));
        }
    }

    get currentFrame(): number {
        return this.runtime.currentFrameTime;
    }

    get playing(): boolean {
        return this.runtime.isAnimationPlaying;
    }

    async play(): Promise<void> {
        await this.runtime.playAnimation();
    }

    pause(): void {
        this.runtime.pauseAnimation();
    }

    async seek(frame: number): Promise<void> {
        await this.runtime.seekAnimation(Math.max(0, frame), true);
    }

    set timeScale(v: number) {
        this.runtime.timeScale = v;
    }

    get timeScale(): number {
        return this.runtime.timeScale;
    }

    get durationFrames(): number {
        return this.runtime.animationFrameTimeDuration;
    }

    dispose(): void {
        this.resizeObserver.disconnect();
        this.unloadModel();
        this.physicsRuntime?.dispose();
        this.engine.dispose();
    }
}

const LINEAR_ROT = [20, 107, 20, 107];
const LINEAR_POS = [20, 107, 20, 107, 20, 107, 20, 107, 20, 107, 20, 107];

/** 合成したキー列 → babylon-mmd の MmdAnimation（VMD を経由せずに直接作る） */
export function toMmdAnimation(motion: ComposedMotion): MmdAnimation {
    const boneTracks: MmdBoneAnimationTrack[] = [];
    const movableTracks: MmdMovableBoneAnimationTrack[] = [];
    for (const b of motion.bones) {
        const n = b.frames.length;
        if (!n) continue;
        const rotInterp = new Uint8Array(n * 4);
        for (let i = 0; i < n; i++) {
            const c = b.curves?.[i];
            // babylon-mmd の並び: [x1, x2, y1, y2]
            rotInterp.set(c ? [c[0], c[2], c[1], c[3]] : LINEAR_ROT, i * 4);
        }
        if (b.movable) {
            const t = new MmdMovableBoneAnimationTrack(b.name, n);
            t.frameNumbers.set(b.frames);
            t.positions.set(b.positions);
            for (let i = 0; i < n; i++) t.positionInterpolations.set(LINEAR_POS, i * 12);
            t.rotations.set(b.rotations);
            t.rotationInterpolations.set(rotInterp);
            t.physicsToggles.fill(1);
            movableTracks.push(t);
        } else {
            const t = new MmdBoneAnimationTrack(b.name, n);
            t.frameNumbers.set(b.frames);
            t.rotations.set(b.rotations);
            t.rotationInterpolations.set(rotInterp);
            t.physicsToggles.fill(1);
            boneTracks.push(t);
        }
    }
    const morphTracks: MmdMorphAnimationTrack[] = [];
    for (const m of motion.morphs) {
        if (!m.frames.length) continue;
        const t = new MmdMorphAnimationTrack(m.name, m.frames.length);
        t.frameNumbers.set(m.frames);
        t.weights.set(m.weights);
        morphTracks.push(t);
    }
    return new MmdAnimation("mmt", boneTracks, movableTracks, morphTracks, new MmdPropertyAnimationTrack(0, []),
        new MmdCameraAnimationTrack(0));
}

export { MMD_FPS };
