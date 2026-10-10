// 動画（<video>）を babylon-mmd の IPlayer にする。MMD のランタイムは、音声と同じようにこの再生時刻に合わせて
// モーションを進めるので、動画とモデルの動きがずれない。
import { Observable } from "@babylonjs/core/Misc/observable";
import type { IPlayer } from "babylon-mmd/esm/Runtime/Audio/IAudioPlayer";

export class VideoPlayer implements IPlayer {
    readonly onDurationChangedObservable = new Observable<void>();
    readonly onPlaybackRateChangedObservable = new Observable<void>();
    readonly onPlayObservable = new Observable<void>();
    readonly onPauseObservable = new Observable<void>();
    readonly onSeekObservable = new Observable<void>();

    private _playbackRate = 1;
    private _metadataLoaded = false;
    private _ignoreSeeked = 0;

    constructor(readonly video: HTMLVideoElement) {
        video.loop = false;
        video.autoplay = false;
        video.preservesPitch = true;
        video.addEventListener("durationchange", this.onDuration);
        video.addEventListener("loadedmetadata", this.onDuration);
        video.addEventListener("playing", this.onPlaying);
        video.addEventListener("pause", this.onPause);
        video.addEventListener("seeked", this.onSeeked);
        if (video.readyState >= 1) this.onDuration();
    }

    dispose(): void {
        const v = this.video;
        v.removeEventListener("durationchange", this.onDuration);
        v.removeEventListener("loadedmetadata", this.onDuration);
        v.removeEventListener("playing", this.onPlaying);
        v.removeEventListener("pause", this.onPause);
        v.removeEventListener("seeked", this.onSeeked);
    }

    private onDuration = () => {
        if (!Number.isFinite(this.video.duration)) return;
        this._metadataLoaded = true;
        this.onDurationChangedObservable.notifyObservers();
    };

    private onPlaying = () => {
        this.video.playbackRate = this._playbackRate;
        this.onPlayObservable.notifyObservers();
    };

    private onPause = () => this.onPauseObservable.notifyObservers();

    private onSeeked = () => {
        if (this._ignoreSeeked > 0) {
            this._ignoreSeeked -= 1;
            return;
        }
        this.onSeekObservable.notifyObservers();
    };

    get duration(): number {
        return Number.isFinite(this.video.duration) ? this.video.duration : 0;
    }

    get currentTime(): number {
        return this.video.currentTime;
    }

    set currentTime(value: number) {
        this.video.currentTime = value;
    }

    _setCurrentTimeWithoutNotify(value: number): void {
        if (Math.abs(this.video.currentTime - value) < 1e-4) return;
        this._ignoreSeeked += 1;
        this.video.currentTime = value;
    }

    get playbackRate(): number {
        return this._playbackRate;
    }

    set playbackRate(value: number) {
        this._setPlaybackRateWithoutNotify(value);
        this.onPlaybackRateChangedObservable.notifyObservers();
    }

    _setPlaybackRateWithoutNotify(value: number): void {
        this._playbackRate = value;
        this.video.playbackRate = value;
    }

    get paused(): boolean {
        return this.video.paused;
    }

    get metadataLoaded(): boolean {
        return this._metadataLoaded;
    }

    async play(): Promise<void> {
        if (this.video.ended || this.video.currentTime >= this.duration - 1e-3) this.video.currentTime = 0;
        await this.video.play();
    }

    pause(): void {
        this.video.pause();
    }
}
